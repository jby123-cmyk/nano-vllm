"""Host-llvm compose test for DecoderLayerStage (decode step)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target.", allow_module_level=True)

from nanovllm.stages.decoder_layer import DecoderLayerStage  # noqa: E402


def test_decoder_layer_decode_compose():
    stage = DecoderLayerStage(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=512,
        tilelang_backend="cpu",
        decode_block_N=64,
        seed=0,
    )
    result = stage.run_decode_step(batch_size=1, context_len=64)
    # Multi-op accumulate; allow rtol matching per-op host gates.
    torch.testing.assert_close(
        result.tilelang_output.float(),
        result.reference_output.float(),
        atol=1e-2,
        rtol=1e-2,
    )
