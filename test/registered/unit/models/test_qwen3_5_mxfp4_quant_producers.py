"""CPU tests for the Qwen3.5 MXFP4 quant producers (ROCm/AITER): selection,
kill switches and fallbacks of the fused Gemma add+RMSNorm+MXFP4 norm and the
dense-MLP SiLU+mul+MXFP4 path. AITER kernels are faked; numerics are covered
by test_quark_mxfp4_asm_producers_gpu.py."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    AsmMxfp4Input,
    QuarkW4A4MXFP4,
)
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_HIDDEN = 512


def _fake_aiter(**ops):
    """sys.modules entries for an ``aiter`` package exposing ``ops`` by dotted
    path, e.g. ``{"ops.rmsnorm.add_rmsnorm_quant": fn}``."""
    modules = {"aiter": types.ModuleType("aiter")}
    for dotted, fn in ops.items():
        *path, attr = dotted.split(".")
        name = "aiter"
        for part in path:
            name = f"{name}.{part}"
            modules.setdefault(name, types.ModuleType(name))
        setattr(modules[name], attr, fn)
    return mock.patch.dict(sys.modules, modules)


def _mxfp4_linear(use_asm=True):
    linear = nn.Module()
    linear.scheme = QuarkW4A4MXFP4({}, {})
    linear.use_aiter_asm_fp4_gemm = use_asm
    return linear


def _on_gfx95():
    return mock.patch.multiple(qwen3_5, _use_aiter=True, _is_gfx95=True, _is_hip=True)


class TestGemmaRMSNormMxfp4Selection(CustomTestCase):
    config = SimpleNamespace(hidden_size=_HIDDEN, rms_norm_eps=1e-6)

    def _norm_for(self, consumer, config=None):
        return qwen3_5._gemma_rms_norm_for(
            config or self.config, consumer, lambda num_tokens: True
        )

    def test_mxfp4_consumer_gets_fused_norm(self):
        with _on_gfx95(), _fake_aiter(**{"ops.rmsnorm.add_rmsnorm_quant": print}):
            norm = self._norm_for(_mxfp4_linear())
        self.assertIsInstance(norm, qwen3_5._GemmaRMSNormMXFP4Quant)

    def test_plain_norm_otherwise(self):
        cases = {
            "kill switch": (_mxfp4_linear(), self.config, True),
            "no consumer (MoE layer)": (None, self.config, False),
            "non-MXFP4 consumer": (nn.Linear(4, 4), self.config, False),
            "hidden % 256": (
                _mxfp4_linear(),
                SimpleNamespace(hidden_size=640, rms_norm_eps=1e-6),
                False,
            ),
        }
        for name, (consumer, config, kill) in cases.items():
            with (
                self.subTest(name),
                _on_gfx95(),
                _fake_aiter(**{"ops.rmsnorm.add_rmsnorm_quant": print}),
                envs.SGLANG_ROCM_DISABLE_NORM_MXFP4_QUANT_FUSION.override(kill),
            ):
                norm = self._norm_for(consumer, config)
            self.assertIs(type(norm), GemmaRMSNorm)

    def test_plain_norm_without_aiter_kernel(self):
        with _on_gfx95(), mock.patch.dict(sys.modules, {"aiter": None}):
            norm = self._norm_for(_mxfp4_linear())
        self.assertIs(type(norm), GemmaRMSNorm)

    def test_consumer_check_follows_asm_flag(self):
        self.assertTrue(qwen3_5._mxfp4_asm_input(_mxfp4_linear(), 4))
        self.assertFalse(qwen3_5._mxfp4_asm_input(_mxfp4_linear(), 0))
        self.assertFalse(qwen3_5._mxfp4_asm_input(_mxfp4_linear(use_asm=False), 4))
        self.assertFalse(qwen3_5._mxfp4_asm_input(None, 4))


class TestGemmaRMSNormMxfp4Forward(CustomTestCase):
    def _norm(self, consumes):
        calls = []

        def add_rmsnorm_quant(*args):
            calls.append(args)

        with _fake_aiter(**{"ops.rmsnorm.add_rmsnorm_quant": add_rmsnorm_quant}):
            norm = qwen3_5._GemmaRMSNormMXFP4Quant(_HIDDEN, 1e-6, consumes)
        norm.weight.data = norm.weight.data.bfloat16()
        return norm, calls

    def _inputs(self, m=3):
        x = torch.randn(m, _HIDDEN, dtype=torch.bfloat16)
        return x, torch.randn_like(x)

    def test_emits_asm_input(self):
        norm, calls = self._norm(lambda num_tokens: True)
        x, residual = self._inputs()
        out, residual_out = norm.forward_hip(x, residual)
        self.assertIsInstance(out, AsmMxfp4Input)
        self.assertEqual(out.x_fp4.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(tuple(out.x_fp4.shape), (3, _HIDDEN // 2))
        self.assertEqual(out.x_scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(out.x_scales.shape), (256, _HIDDEN // 32))
        self.assertEqual(residual_out.shape, x.shape)
        (args,) = calls
        # group_size=32, shuffle_scale=True, gemma_norm=True
        self.assertEqual(args[-3:], (32, True, True))
        self.assertIs(args[1], x)
        self.assertIs(args[2], residual)

    def test_falls_back_to_bf16_norm(self):
        x, residual = self._inputs()
        cases = {
            "first layer (no residual)": ((x,), True),
            "post-residual addition": ((x, residual, torch.zeros_like(x)), True),
            "consumer declines the batch": ((x, residual), False),
        }
        for name, (args, consumes) in cases.items():
            with self.subTest(name):
                norm, calls = self._norm(lambda num_tokens, c=consumes: c)
                out = norm.forward_hip(*args)
                out = out[0] if isinstance(out, tuple) else out
                self.assertIsInstance(out, torch.Tensor)
                self.assertEqual(calls, [])


class TestSiluMxfp4Producer(CustomTestCase):
    def _mlp(self, down_use_asm=True):
        return SimpleNamespace(
            gate_up_proj=_mxfp4_linear(),
            down_proj=_mxfp4_linear(down_use_asm),
            _enable_silu_mxfp4_quant_fusion=False,
        )

    def _enable(self, mlp):
        with (
            _on_gfx95(),
            _fake_aiter(**{"ops.triton.activation.act_mul_and_mxfp4_quant": print}),
        ):
            qwen3_5._maybe_enable_silu_mxfp4_quant_fusion(mlp)
        return mlp._enable_silu_mxfp4_quant_fusion

    def test_enabled_for_mxfp4_mlp(self):
        self.assertTrue(self._enable(self._mlp()))

    def test_kill_switch_and_non_mxfp4(self):
        with envs.SGLANG_DISABLE_SILU_MXFP4_QUANT_FUSION.override(True):
            self.assertFalse(self._enable(self._mlp()))
        mlp = self._mlp()
        mlp.down_proj = nn.Linear(4, 4)
        self.assertFalse(self._enable(mlp))

    def test_fused_path_needs_asm_down_proj(self):
        gate_up = torch.zeros(3, 64, dtype=torch.bfloat16)
        mlp = self._mlp()
        mlp._enable_silu_mxfp4_quant_fusion = True
        self.assertTrue(Qwen2MoeMLP._silu_mxfp4_quant_applies(mlp, gate_up))
        self.assertFalse(Qwen2MoeMLP._silu_mxfp4_quant_applies(mlp, gate_up[:0]))
        self.assertFalse(Qwen2MoeMLP._silu_mxfp4_quant_applies(mlp, (gate_up,)))
        mlp.down_proj.use_aiter_asm_fp4_gemm = False
        self.assertFalse(Qwen2MoeMLP._silu_mxfp4_quant_applies(mlp, gate_up))

    def test_producer_requests_roundup_and_shuffled_scales(self):
        calls = []

        def act_mul_and_mxfp4_quant(x, activation, **kwargs):
            calls.append((activation, kwargs))
            m, n = x.shape
            return (
                torch.zeros(m, n // 4, dtype=torch.uint8),
                torch.zeros(256, n // 64, dtype=torch.uint8),
            )

        gate_up = torch.zeros(3, 1024, dtype=torch.bfloat16)
        with _fake_aiter(
            **{"ops.triton.activation.act_mul_and_mxfp4_quant": act_mul_and_mxfp4_quant}
        ):
            out = Qwen2MoeMLP._silu_mxfp4_quant_fused(None, gate_up)
        self.assertIsInstance(out, AsmMxfp4Input)
        self.assertEqual(out.x_fp4.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(out.x_scales.dtype, torch.float8_e8m0fnu)
        self.assertEqual(calls, [("silu", {"scaling_mode": "ceil", "shuffle": True})])


class TestFusedArInputSelection(CustomTestCase):
    def test_marker_only_reaches_asm_linears(self):
        marker = AsmMxfp4Input(torch.zeros(1, 1), torch.zeros(1, 1))
        linear = _mxfp4_linear()
        self.assertIs(qwen3_5._select_fused_ar_input_for_linear(marker, linear), marker)
        with self.assertRaises(TypeError):
            qwen3_5._select_fused_ar_input_for_linear(
                marker, _mxfp4_linear(use_asm=False)
            )


if __name__ == "__main__":
    unittest.main()
