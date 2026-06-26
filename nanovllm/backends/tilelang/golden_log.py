"""Per-build golden-reference comparison logs for demo runs."""

from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Any

import torch

GOLDEN_LOG_NAME = "golden.log"


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def golden_log_path(build_dir: str) -> str:
    return os.path.join(build_dir, GOLDEN_LOG_NAME)


def _format_block(title: str, body: str) -> str:
    return f"{title}\n{'-' * len(title)}\n{body.rstrip()}\n"


def format_golden_log(
    *,
    demo: str,
    status: str,
    reference: str,
    backend: str,
    passed: bool | None,
    max_abs_diff: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    output_shape: tuple | list | None = None,
    error: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        f"Golden comparison log",
        f"Generated UTC: {now}",
        "",
        _format_block(
            "Run",
            "\n".join(
                [
                    f"demo      : {demo}",
                    f"status    : {status}",
                    f"reference : {reference}",
                    f"backend   : {backend}",
                ]
            ),
        ),
    ]

    if passed is not None:
        lines.append(_format_block("Result", f"passed         : {passed}"))
    if max_abs_diff is not None or output_shape is not None or atol is not None:
        metric_lines = []
        if output_shape is not None:
            metric_lines.append(f"output shape   : {tuple(output_shape)}")
        if max_abs_diff is not None:
            metric_lines.append(f"max abs diff   : {max_abs_diff}")
        if atol is not None:
            metric_lines.append(f"atol           : {atol}")
        if rtol is not None:
            metric_lines.append(f"rtol           : {rtol}")
        lines.append(_format_block("Metrics", "\n".join(metric_lines)))

    if details:
        detail_lines = [f"{key}: {value}" for key, value in details.items()]
        lines.append(_format_block("Details", "\n".join(detail_lines)))

    if error:
        lines.append(_format_block("Error", error))

    return "\n".join(lines).rstrip() + "\n"


def write_golden_log(build_dir: str, text: str) -> str:
    path = golden_log_path(build_dir)
    _write_text(path, text)
    return path


def write_skipped_golden_log(
    build_dir: str,
    *,
    demo: str,
    reference: str,
    backend: str,
    reason: str,
    details: dict[str, Any] | None = None,
) -> str:
    text = format_golden_log(
        demo=demo,
        status="skipped",
        reference=reference,
        backend=backend,
        passed=None,
        error=reason,
        details=details,
    )
    return write_golden_log(build_dir, text)


def compare_and_log(
    build_dir: str,
    *,
    demo: str,
    reference: str,
    backend: str,
    tilelang_tensor: torch.Tensor,
    reference_tensor: torch.Tensor,
    atol: float,
    rtol: float,
    max_abs_diff: float | None = None,
    details: dict[str, Any] | None = None,
    raise_on_fail: bool = True,
) -> tuple[str, bool]:
    """Compare TileLang output to the golden reference and write ``golden.log``.

    Returns ``(log_path, passed)``. Re-raises after logging when ``raise_on_fail``.
    """
    if max_abs_diff is None:
        max_abs_diff = (reference_tensor.float() - tilelang_tensor.float()).abs().max().item()

    try:
        torch.testing.assert_close(
            tilelang_tensor.float(),
            reference_tensor.float(),
            atol=atol,
            rtol=rtol,
        )
        text = format_golden_log(
            demo=demo,
            status="pass",
            reference=reference,
            backend=backend,
            passed=True,
            max_abs_diff=max_abs_diff,
            atol=atol,
            rtol=rtol,
            output_shape=tuple(reference_tensor.shape),
            details=details,
        )
        return write_golden_log(build_dir, text), True
    except Exception as exc:  # noqa: BLE001
        err_lines = traceback.format_exc().strip().splitlines()
        one_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
        text = format_golden_log(
            demo=demo,
            status="fail",
            reference=reference,
            backend=backend,
            passed=False,
            max_abs_diff=max_abs_diff,
            atol=atol,
            rtol=rtol,
            output_shape=tuple(reference_tensor.shape),
            error=f"{one_line}\n\n{chr(10).join(err_lines[-12:])}",
            details=details,
        )
        path = write_golden_log(build_dir, text)
        if raise_on_fail:
            raise
        return path, False


def log_numeric_exception(
    build_dir: str,
    *,
    demo: str,
    reference: str,
    backend: str,
    exc: Exception,
    details: dict[str, Any] | None = None,
) -> str:
    err_lines = traceback.format_exc().strip().splitlines()
    one_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
    text = format_golden_log(
        demo=demo,
        status="fail",
        reference=reference,
        backend=backend,
        passed=False,
        error=f"{one_line}\n\n{chr(10).join(err_lines[-12:])}",
        details=details,
    )
    return write_golden_log(build_dir, text)
