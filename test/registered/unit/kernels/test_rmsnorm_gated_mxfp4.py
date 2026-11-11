"""CPU-only policy tests for fused gated RMSNorm MXFP4."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.rmsnorm_gated_mxfp4 import (
    _is_mxfp4_shuffled_width_supported,
    _is_rmsnorm_gated_mxfp4_layout_supported,
    mxfp4_scale_shape,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRMSNormGatedMXFP4Policy(unittest.TestCase):
    def test_scale_shapes_match_both_linear_consumers(self):
        self.assertEqual(mxfp4_scale_shape(4, 192, shuffle=False), (4, 192))
        self.assertEqual(mxfp4_scale_shape(4, 192, shuffle=True), (256, 192))
        self.assertEqual(mxfp4_scale_shape(257, 193, shuffle=True), (512, 200))

    def test_shuffled_activation_width_alignment(self):
        self.assertTrue(_is_mxfp4_shuffled_width_supported(6144))
        self.assertTrue(_is_mxfp4_shuffled_width_supported(768))
        self.assertFalse(_is_mxfp4_shuffled_width_supported(288))
        self.assertFalse(_is_mxfp4_shuffled_width_supported(128))

    def test_qwen38_strided_prefill_layout_is_supported(self):
        num_tokens, num_heads, head_dim = 4, 48, 128
        x_storage = torch.empty(
            (num_tokens * num_heads, head_dim + 32), dtype=torch.bfloat16
        )
        z_storage = torch.empty(
            (num_tokens, num_heads, head_dim * 2), dtype=torch.bfloat16
        )
        x = x_storage[:, :head_dim]
        z = z_storage[..., head_dim:]
        weight = torch.empty(head_dim, dtype=torch.bfloat16)

        self.assertTrue(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=num_heads,
                activation="swish",
                shuffle_scales=True,
            )
        )

    def test_unaligned_width_only_rejects_shuffled_layout(self):
        x = torch.empty((3, 96), dtype=torch.bfloat16)
        z = torch.empty((1, 3, 96), dtype=torch.bfloat16)
        weight = torch.empty(96, dtype=torch.bfloat16)

        self.assertTrue(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=3,
                activation="swish",
                shuffle_scales=False,
            )
        )
        self.assertFalse(
            _is_rmsnorm_gated_mxfp4_layout_supported(
                x,
                z,
                weight,
                num_heads=3,
                activation="swish",
                shuffle_scales=True,
            )
        )

    def test_incompatible_layouts_use_fallback(self):
        x = torch.empty((8, 128), dtype=torch.bfloat16)
        z = torch.empty((2, 4, 128), dtype=torch.bfloat16)
        weight = torch.empty(128, dtype=torch.bfloat16)

        cases = (
            (x[:0], z[:0], weight, 4, "swish"),
            (x, z, weight, 3, "swish"),
            (x, z[..., ::2], weight, 4, "swish"),
            (x, z, weight, 4, "gelu"),
            (x.to(torch.float32), z.to(torch.float32), weight.float(), 4, "swish"),
        )
        for case in cases:
            with self.subTest(case=case[3:]):
                self.assertFalse(
                    _is_rmsnorm_gated_mxfp4_layout_supported(
                        case[0],
                        case[1],
                        case[2],
                        num_heads=case[3],
                        activation=case[4],
                    )
                )


if __name__ == "__main__":
    unittest.main()
