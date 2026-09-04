"""Regression tests for the GDN conv1d weight cache refresh.

The Ascend GDN backend caches a transposed copy of conv1d.weight on the
RadixLinearAttention wrapper (AscendGDNAttnBackend._get_conv_weights_t).
mamba_v2_sharded_weight_loader writes through ``param.data[...] = ...``;
because ``Tensor.data`` carries a version counter independent of the
Parameter's, online weight updates (e.g. RL Actor->SGLang sync via
``update_weights_from_tensor``) are invisible to ``_version``-based
invalidation. conv1d.weight's loader is therefore wrapped with
``wrap_conv1d_weight_loader`` to refresh the cached copy explicitly, in
place, so captured graphs keep referencing the same storage.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.radix_linear_attention import wrap_conv1d_weight_loader

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CHANNELS = 16  # q(4) + k(4) + v(8) on a single TP rank
_KERNEL = 4


def _make_conv_param() -> torch.nn.Parameter:
    # conv1d.weight layout: [channels, 1, kernel]
    return torch.nn.Parameter(
        torch.randn(_CHANNELS, 1, _KERNEL), requires_grad=False
    )


def _make_base_loader():
    # Same shard layout as Qwen3_5: q/k shards then the v shard, TP size 1.
    return mamba_v2_sharded_weight_loader(
        [(4, 0, False), (4, 0, False), (8, 0, False)], 1, 0
    )


class _FakeAttn:
    """Stands in for RadixLinearAttention: holds conv_weights and the cache."""

    def __init__(self, conv_param: torch.nn.Parameter):
        self.conv_weights = conv_param.view(_CHANNELS, _KERNEL)
        self._conv_weights_t = self.conv_weights.transpose(0, 1).contiguous()


class TestConvWeightCacheRefresh(CustomTestCase):

    def test_mamba_loader_write_bypasses_param_version(self):
        """Root-cause guard: the real loader must not bump ``_version``.

        If a future PyTorch/SGLang change makes param.data writes bump the
        shared version counter, this test fails and the explicit wrapper may
        be revisited.
        """
        param = _make_conv_param()
        view = param.view(_CHANNELS, _KERNEL)
        param_version = param._version
        view_version = view._version

        _make_base_loader()(param, torch.randn_like(param))

        self.assertEqual(param._version, param_version)
        self.assertEqual(view._version, view_version)

    def test_wrapped_loader_refreshes_cache_in_place(self):
        """An online reload must refresh the cached transpose in place."""
        param = _make_conv_param()
        attn = _FakeAttn(param)
        cached = attn._conv_weights_t
        cached_ptr = cached.data_ptr()

        new_weight = torch.randn_like(param)
        wrapped = wrap_conv1d_weight_loader(attn, _make_base_loader())
        wrapped(param, new_weight)

        # The loader actually wrote the new weights.
        torch.testing.assert_close(param.detach(), new_weight, rtol=0, atol=0)
        # The cache is refreshed in place: same tensor object, same storage
        # (captured graphs keep referencing it), content matches the
        # transpose of the new weights.
        self.assertIs(attn._conv_weights_t, cached)
        self.assertEqual(attn._conv_weights_t.data_ptr(), cached_ptr)
        reference = param.view(_CHANNELS, _KERNEL).transpose(0, 1)
        torch.testing.assert_close(attn._conv_weights_t, reference, rtol=0, atol=0)

    def test_wrapped_loader_without_cache_is_noop(self):
        """Initial load (before the first forward) must not create a cache."""
        param = _make_conv_param()
        attn = _FakeAttn(param)
        del attn._conv_weights_t

        new_weight = torch.randn_like(param)
        wrapped = wrap_conv1d_weight_loader(attn, _make_base_loader())
        wrapped(param, new_weight)

        self.assertFalse(hasattr(attn, "_conv_weights_t"))
        torch.testing.assert_close(param.detach(), new_weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
