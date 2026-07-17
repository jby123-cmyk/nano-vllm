"""
RVV TileLang inference example — mirrors ``example.py`` on the RVV decode path.

Setup (from ``usage.md``)::

    conda activate nanovllm
    export PYTHONPATH=/mnt/ssd/jby123/tilelang:/mnt/ssd/jby123/tilelang/build/python:$PYTHONPATH
    export LD_LIBRARY_PATH=/mnt/ssd/jby123/miniconda3/envs/nanovllm/lib:$LD_LIBRARY_PATH
    cd /mnt/ssd/jby123/nano-vllm

On a non-RISC-V host (cross-compile workstation), ``device="rvv"`` defaults to
``rvv_compile_only=True``: kernels are lowered to ``demos/build_rvv/engine/``
during ``LLM`` init and ``generate()`` is skipped. On an RVV host, set
``rvv_compile_only=False`` to execute the full decode loop via native TVM FFI
(Spike session integration deferred).
"""

import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        device="rvv",
        enforce_eager=True,
        tensor_parallel_size=1,
        rvv_nr_lanes=4,
        rvv_build_dir="demos/build_rvv/engine",
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    if llm.model_runner.config.rvv_compile_only:
        manifest = os.path.join(llm.model_runner.config.rvv_build_dir, "engine_kernels.json")
        print(
            "RVV compile-only mode: TileLang kernels lowered during init.\n"
            f"  build_dir : {llm.model_runner.config.rvv_build_dir}\n"
            f"  manifest  : {manifest}\n"
            "Set rvv_compile_only=False on an RVV host to run generate()."
        )
        return

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
