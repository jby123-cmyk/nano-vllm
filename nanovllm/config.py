import os
from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.engine.device import default_rvv_compile_only


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # Execution device: "cuda" (default) or "rvv" (TileLang RVV decode path).
    device: str = "cuda"
    # TileLang codegen/execution target passed to ``run_tilelang_*``.
    tilelang_backend: str = "cuda"
    # RVV target VLEN lanes (usage.md: VLEN = 1024 * nr_lanes bits per cluster).
    rvv_nr_lanes: int = 4
    # Artifact root for ``lower_kernel_rvv`` when ``device="rvv"``.
    rvv_build_dir: str = "demos/build_rvv/engine"
    # Fixed KV pool size on host when ``device="rvv"`` (CUDA mem probe unused).
    rvv_num_kvcache_blocks: int = 64
    # Cross-compile only (no kernel execution). Defaults True on non-riscv64 hosts.
    rvv_compile_only: bool | None = None
    # RVV execution path when ``device="rvv"``: compile_only | spike | native.
    rvv_execution_backend: str = "compile_only"
    # Host llvm TileLang reference on CPU (parity baseline; not Spike/RVV execution).
    rvv_host_cpu_tilelang: bool = False
    # One prefill token per scheduler step (chunked prefill); avoids shape bucket explosion.
    rvv_serial_prefill: bool = False
    # Write ``generate_report.json`` + ``generate_report.md`` under ``rvv_build_dir``.
    rvv_generate_report: bool | None = None
    # Per-kernel host llvm replay + max_abs_diff on Spike runs.
    rvv_generate_report_compare: bool = False
    # Save per-step logits trace (reference runs) for stage-level diffs.
    rvv_generate_report_trace: bool = False
    # Optional reference build dir with ``golden_trace.jsonl`` + golden tokens.
    rvv_generate_report_golden_dir: str | None = None
    # Decode attention backend: "flash_attn" (default) or "tilelang"
    # (paged TileLang kernel matching flash_attn_with_kvcache).
    attn_backend: str = "flash_attn"
    # Linear (F.linear / GEMM) backend: "torch" (default) or "tilelang".
    linear_backend: str = "torch"
    # RMSNorm backend: "torch" (default) or "tilelang".
    norm_backend: str = "torch"
    # SiluAndMul backend: "torch" (default) or "tilelang".
    act_backend: str = "torch"
    # RoPE apply backend: "torch" (default) or "tilelang".
    rope_backend: str = "torch"
    # Embedding gather backend: "torch" (default) or "tilelang".
    embed_backend: str = "torch"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert self.device in ("cuda", "rvv"), (
            f"device must be 'cuda' or 'rvv', got {self.device!r}"
        )
        assert self.tilelang_backend in ("cuda", "cpu", "rvv"), (
            f"tilelang_backend must be 'cuda', 'cpu', or 'rvv', got {self.tilelang_backend!r}"
        )
        assert self.rvv_execution_backend in ("compile_only", "spike", "native"), (
            "rvv_execution_backend must be 'compile_only', 'spike', or 'native', "
            f"got {self.rvv_execution_backend!r}"
        )
        if self.device == "rvv":
            assert self.tensor_parallel_size == 1, (
                "device='rvv' requires tensor_parallel_size=1 (no RVV all_reduce yet)"
            )
            self.enforce_eager = True
            self.attn_backend = "tilelang"
            self.linear_backend = "tilelang"
            self.norm_backend = "tilelang"
            self.act_backend = "tilelang"
            self.rope_backend = "tilelang"
            self.embed_backend = "tilelang"
            if self.rvv_host_cpu_tilelang:
                self.tilelang_backend = "cpu"
                self.rvv_compile_only = False
                self.rvv_execution_backend = "compile_only"
            else:
                self.tilelang_backend = "rvv"
            if self.rvv_execution_backend == "spike":
                self.rvv_compile_only = False
            elif self.rvv_compile_only is None:
                self.rvv_compile_only = default_rvv_compile_only()
            if self.num_kvcache_blocks < 0:
                self.num_kvcache_blocks = self.rvv_num_kvcache_blocks
            if self.rvv_serial_prefill:
                self.max_num_batched_tokens = 1
            if self.rvv_generate_report is None:
                self.rvv_generate_report = self.rvv_execution_backend == "spike"
        else:
            if self.rvv_compile_only is None:
                self.rvv_compile_only = False
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.attn_backend in ("flash_attn", "tilelang"), (
            f"attn_backend must be 'flash_attn' or 'tilelang', got {self.attn_backend!r}"
        )
        assert self.linear_backend in ("torch", "tilelang"), (
            f"linear_backend must be 'torch' or 'tilelang', got {self.linear_backend!r}"
        )
        assert self.norm_backend in ("torch", "tilelang"), (
            f"norm_backend must be 'torch' or 'tilelang', got {self.norm_backend!r}"
        )
        assert self.act_backend in ("torch", "tilelang"), (
            f"act_backend must be 'torch' or 'tilelang', got {self.act_backend!r}"
        )
        assert self.rope_backend in ("torch", "tilelang"), (
            f"rope_backend must be 'torch' or 'tilelang', got {self.rope_backend!r}"
        )
        assert self.embed_backend in ("torch", "tilelang"), (
            f"embed_backend must be 'torch' or 'tilelang', got {self.embed_backend!r}"
        )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
