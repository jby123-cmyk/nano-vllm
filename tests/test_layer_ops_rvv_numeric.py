"""Host-llvm numeric tests for remaining TileLang layer ops."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tilelang")

from tilelang import tvm as _tvm  # noqa: E402

if not _tvm.runtime.enabled("llvm"):
    pytest.skip("TVM build has no LLVM target.", allow_module_level=True)

ATOL = 1e-2
RTOL = 1e-2


def test_rmsnorm_plain():
    from nanovllm.stages.rmsnorm import RMSNormStage

    r = RMSNormStage(hidden=128, fuse_residual=False, tilelang_backend="cpu").run(16)
    assert r.max_abs_diff <= ATOL, r.max_abs_diff


def test_rmsnorm_fused_residual():
    from nanovllm.stages.rmsnorm import RMSNormStage

    r = RMSNormStage(hidden=128, fuse_residual=True, tilelang_backend="cpu").run(16)
    assert r.max_abs_diff <= ATOL, r.max_abs_diff
    assert r.residual_max_abs_diff is not None and r.residual_max_abs_diff <= ATOL


def test_silu_mul():
    from nanovllm.stages.activation import SiluMulStage

    r = SiluMulStage(intermediate=256, tilelang_backend="cpu").run(8)
    assert r.max_abs_diff <= ATOL, r.max_abs_diff


def test_rope():
    from nanovllm.stages.rope import RopeStage

    r = RopeStage(
        num_heads=8, num_kv_heads=2, head_dim=64, tilelang_backend="cpu"
    ).run(4)
    assert r.max_abs_diff <= ATOL, r.max_abs_diff


def test_kv_store():
    from nanovllm.backends.tilelang.kv_store import run_tilelang_store_kvcache

    N, H, D = 4, 2, 64
    key = torch.randn(N, H, D)
    value = torch.randn(N, H, D)
    num_slots = 16
    k_cache = torch.zeros(num_slots, H * D)
    v_cache = torch.zeros_like(k_cache)
    # View as engine layout [slots, 1, H*D] style — stride(1)==D for 3D:
    k3 = k_cache.view(num_slots, H, D)
    v3 = v_cache.view(num_slots, H, D)
    slots = torch.tensor([3, 7, -1, 1], dtype=torch.int32)
    run_tilelang_store_kvcache(key, value, k3, v3, slots, backend="cpu")
    for i, s in enumerate(slots.tolist()):
        if s < 0:
            continue
        torch.testing.assert_close(k3[s], key[i], atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v3[s], value[i], atol=ATOL, rtol=RTOL)


def test_embedding():
    from nanovllm.backends.tilelang.embedding import run_tilelang_embedding

    vocab, hidden, n = 128, 64, 8
    weight = torch.randn(vocab, hidden)
    ids = torch.randint(0, vocab, (n,), dtype=torch.int32)
    ref = weight[ids.long()]
    out = run_tilelang_embedding(ids, weight, backend="cpu")
    torch.testing.assert_close(out, ref, atol=ATOL, rtol=RTOL)
