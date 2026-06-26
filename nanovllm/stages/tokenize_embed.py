from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from nanovllm.backends.tilelang.build_dump import dump_embedding_build
from nanovllm.backends.tilelang.embedding import dump_tensorir, run_tilelang_embedding
from nanovllm.backends.tilelang.weights import load_embedding_weight, load_model_dims


@dataclass
class TokenizeEmbedResult:
    prompt: str
    token_ids: list[int]
    input_ids: torch.Tensor
    reference_hidden: torch.Tensor
    tilelang_hidden: torch.Tensor
    max_abs_diff: float


class TokenizeEmbedStage:
    """
    Tokenization + embedding only — the first part of the nano-vllm engine path.

    Mirrors:
      - llm_engine.LLMEngine.add_request()     → tokenize()
      - model_runner.ModelRunner.prepare_prefill() → prepare_input_ids()
      - qwen3.Qwen3Model.forward() embed step  → embed_reference()
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        random_weights: bool = False,
        vocab_override: int | None = None,
        hidden_override: int | None = None,
        token_block: int = 1,
        threads: int = 256,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TokenizeEmbedStage.")

        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.random_weights = random_weights
        self.token_block = token_block
        self.threads = threads

        self.vocab, self.hidden = load_model_dims(
            model_path, vocab_override, hidden_override
        )

        if random_weights:
            self.weight = torch.randn(
                self.vocab, self.hidden, device=device, dtype=dtype
            )
            self.tokenizer = None
            self.hf_config = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            self.weight = load_embedding_weight(
                model_path, self.vocab, self.hidden, dtype, device
            )
            self.hf_config = AutoConfig.from_pretrained(model_path)

    def tokenize(self, prompt: str) -> list[int]:
        """Same entry as LLMEngine.add_request() for string prompts."""
        if self.random_weights:
            raise RuntimeError(
                "random_weights=True skips tokenizer setup. "
                "Pass --prompt-token-ids or disable --random-weights."
            )
        return self.tokenizer.encode(prompt)

    def prepare_input_ids(self, token_ids: list[int]) -> torch.Tensor:
        """
        Build the GPU tensor passed into Qwen3Model.forward() during prefill.

        Matches model_runner.prepare_prefill() layout for a single sequence
        with no prefix cache: a flat 1D int64 tensor on CUDA.
        """
        return torch.tensor(token_ids, dtype=torch.int64, device=self.device)

    def embed_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Original engine path for tp_size=1.

        Equivalent to VocabParallelEmbedding.forward() when tensor parallel is
        disabled: F.embedding(input_ids, weight).
        """
        return F.embedding(input_ids, self.weight)

    def embed_tilelang(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compiled TileLang backend for the same embedding gather."""
        return run_tilelang_embedding(
            input_ids,
            self.weight,
            token_block=self.token_block,
            threads=self.threads,
        )

    def dump_tilelang_tir(self, num_tokens: int, dump_path: str = "") -> str:
        in_dtype = "float16" if self.dtype == torch.float16 else "float32"
        return dump_tensorir(
            num_tokens,
            self.hidden,
            self.vocab,
            self.token_block,
            self.threads,
            in_dtype,
            dump_path,
        )

    def dump_build_artifacts(
        self,
        num_tokens: int,
        build_dir: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        in_dtype = "float16" if self.dtype == torch.float16 else "float32"
        return dump_embedding_build(
            num_tokens,
            self.hidden,
            self.vocab,
            self.token_block,
            self.threads,
            in_dtype,
            build_dir=build_dir,
            metadata=metadata,
        )

    def run(
        self,
        prompt: str | None = None,
        token_ids: list[int] | None = None,
        dump_tir_path: str = "",
        dump_build_dir: str | None = None,
    ) -> TokenizeEmbedResult:
        if (prompt is None) == (token_ids is None):
            raise ValueError("Pass exactly one of prompt or token_ids.")

        if prompt is not None:
            token_ids = self.tokenize(prompt)
        prompt_text = prompt or "<token_ids>"

        input_ids = self.prepare_input_ids(token_ids)

        run_meta = {
            "prompt": prompt_text,
            "token_ids": token_ids,
            "model_path": self.model_path,
            "random_weights": self.random_weights,
        }
        if dump_build_dir is not None:
            self.dump_build_artifacts(
                input_ids.numel(),
                build_dir=dump_build_dir,
                metadata=run_meta,
            )
        elif dump_tir_path:
            self.dump_tilelang_tir(input_ids.numel(), dump_tir_path)

        reference_hidden = self.embed_reference(input_ids)
        tilelang_hidden = self.embed_tilelang(input_ids)
        max_abs_diff = (reference_hidden - tilelang_hidden).abs().max().item()

        return TokenizeEmbedResult(
            prompt=prompt_text,
            token_ids=token_ids,
            input_ids=input_ids,
            reference_hidden=reference_hidden,
            tilelang_hidden=tilelang_hidden,
            max_abs_diff=max_abs_diff,
        )
