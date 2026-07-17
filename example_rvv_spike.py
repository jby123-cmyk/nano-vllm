"""
Tiny 2-layer Qwen fixture + Spike ``generate()`` demo (Phase 4).

Build the fixture once (weights are gitignored)::

    python scripts/build_tiny_qwen_fixture.py --num-layers 2

Then run::

    python example_rvv_spike.py
"""

from __future__ import annotations

import json
import os

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.backends.spike.engine_golden import ensure_tiny_fixture_2l, tiny_llm_kwargs


def main() -> None:
    fixture = ensure_tiny_fixture_2l()
    meta_path = os.path.join(fixture, "fixture_meta.json")
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    prompt_tokens = meta["prompt_tokens"]

    torch.manual_seed(0)
    llm_kwargs = tiny_llm_kwargs(spike=True, num_layers=2, serial_prefill=True)
    llm = LLM(
        fixture,
        **llm_kwargs,
    )
    try:
        sampling_params = SamplingParams(temperature=1e-9, max_tokens=8)
        outputs = llm.generate([prompt_tokens], sampling_params, use_tqdm=True)
        for output in outputs:
            print(f"token_ids: {output['token_ids']!r}")
            print(f"text: {output['text']!r}")
        report_path = os.path.join(llm_kwargs["rvv_build_dir"], "generate_report.json")
        print(f"generate_report: {report_path}")
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
