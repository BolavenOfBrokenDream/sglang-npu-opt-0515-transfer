"""Unit tests for Sampler._attach_logprobs_to_output.

Regression coverage for two bugs in the decode logprob attach path:
  - iterating ``logits_output.token_ids_logprobs_val`` (a nonexistent field on
    LogitsProcessorOutput) instead of ``next_token_token_ids_logprobs_val``
    raised AttributeError for any request with specific token-ID logprobs;
  - the top-logprob branch ran a full-vocab topk twice (get_top_logprobs plus
    an immediate re-topk that overwrote the first result).

The method under test does not touch ``self``, so the Sampler is allocated
with ``__new__`` to skip the distributed-group init in ``__init__``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_VOCAB = 1000


def _make_sampler() -> Sampler:
    # _attach_logprobs_to_output never reads instance state; bypass __init__
    # (which requires an initialized torch distributed group).
    return Sampler.__new__(Sampler)


def _attach(
    logprobs: torch.Tensor,
    top_logprobs_nums,
    token_ids_logprobs,
    batch_next_token_ids: torch.Tensor,
    logprobs_are_probs: bool = False,
) -> LogitsProcessorOutput:
    logits_output = LogitsProcessorOutput(next_token_logits=None)
    _make_sampler()._attach_logprobs_to_output(
        logits_output=logits_output,
        logprobs=logprobs,
        top_logprobs_nums=top_logprobs_nums,
        token_ids_logprobs=token_ids_logprobs,
        sampling_info=None,
        batch_next_token_ids=batch_next_token_ids,
        logprobs_are_probs=logprobs_are_probs,
    )
    return logits_output


class TestAttachTokenIdsLogprobs(CustomTestCase):

    def test_token_ids_logprobs_no_attribute_error(self):
        """Specific token-ID logprob requests must not raise AttributeError."""
        logprobs = torch.log_softmax(torch.randn(2, _VOCAB), dim=-1)
        out = _attach(
            logprobs=logprobs.clone(),
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[[5, 7], [42]],
            batch_next_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        )
        self.assertIsNotNone(out.next_token_token_ids_logprobs_val)
        self.assertIsNotNone(out.next_token_token_ids_logprobs_idx)

    def test_token_ids_logprobs_values(self):
        logprobs = torch.log_softmax(torch.randn(3, _VOCAB), dim=-1)
        reference = logprobs.clone()
        requested = [[5, 7], None, [42]]
        out = _attach(
            logprobs=logprobs.clone(),
            top_logprobs_nums=[0, 0, 0],
            token_ids_logprobs=requested,
            batch_next_token_ids=torch.tensor([3, 4, 9], dtype=torch.int32),
        )
        vals = out.next_token_token_ids_logprobs_val
        idxs = out.next_token_token_ids_logprobs_idx
        self.assertEqual(len(vals), 3)
        torch.testing.assert_close(
            vals[0], reference[0, torch.tensor([5, 7])], rtol=0, atol=0
        )
        self.assertEqual(idxs[0], [5, 7])
        # None request yields empty placeholders.
        self.assertEqual(vals[1], [])
        self.assertEqual(idxs[1], [])
        torch.testing.assert_close(
            vals[2], reference[2, torch.tensor([42])], rtol=0, atol=0
        )
        self.assertEqual(idxs[2], [42])

    def test_token_ids_logprobs_probs_input(self):
        """logprobs_are_probs=True applies log() before clamping, per row."""
        probs = torch.softmax(torch.randn(2, _VOCAB), dim=-1)
        reference = probs.clone()
        out = _attach(
            logprobs=probs.clone(),
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[[1, 2], [3]],
            batch_next_token_ids=torch.tensor([0, 1], dtype=torch.int32),
            logprobs_are_probs=True,
        )
        torch.testing.assert_close(
            out.next_token_token_ids_logprobs_val[0],
            reference[0, torch.tensor([1, 2])].log(),
            rtol=1e-6,
            atol=1e-6,
        )


class TestAttachTopLogprobs(CustomTestCase):

    def test_top_logprobs_values(self):
        """Per-request k slicing stays correct with the single-topk path."""
        logprobs = torch.log_softmax(torch.randn(2, _VOCAB), dim=-1)
        reference = logprobs.clone()
        out = _attach(
            logprobs=logprobs.clone(),
            top_logprobs_nums=[3, 1],
            token_ids_logprobs=[None, None],
            batch_next_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        )
        ref_vals, ref_idx = reference.topk(3, dim=-1)
        self.assertEqual(len(out.next_token_top_logprobs_val), 2)
        torch.testing.assert_close(
            out.next_token_top_logprobs_val[0], ref_vals[0], rtol=0, atol=0
        )
        self.assertTrue(torch.equal(out.next_token_top_logprobs_idx[0], ref_idx[0]))
        torch.testing.assert_close(
            out.next_token_top_logprobs_val[1], ref_vals[1][:1], rtol=0, atol=0
        )
        self.assertTrue(torch.equal(out.next_token_top_logprobs_idx[1], ref_idx[1][:1]))

    def test_next_token_logprobs_gather(self):
        logprobs = torch.log_softmax(torch.randn(2, _VOCAB), dim=-1)
        reference = logprobs.clone()
        out = _attach(
            logprobs=logprobs.clone(),
            top_logprobs_nums=[0, 0],
            token_ids_logprobs=[None, None],
            batch_next_token_ids=torch.tensor([3, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            out.next_token_logprobs,
            torch.stack([reference[0, 3], reference[1, 4]]),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
