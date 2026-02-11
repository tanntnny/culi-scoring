from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from src.utils.logging import ensure_dir


def run(cfg: DictConfig) -> None:
    run_dir = Path(getattr(cfg, "run_dir", Path.cwd()))
    reports_dir = Path(cfg.paths.reports_dir)
    ensure_dir(reports_dir)

    metrics_path = run_dir / "metrics.json"
    eval_metrics_path = run_dir / "metrics_eval.json"

    report_lines = ["Run Report", "", f"Run directory: {run_dir}"]
    if metrics_path.exists():
        report_lines.append(f"Train metrics: {metrics_path}")
    if eval_metrics_path.exists():
        report_lines.append(f"Eval metrics: {eval_metrics_path}")

    report_path = reports_dir / "report.txt"
    report_path.write_text("\n".join(report_lines))
