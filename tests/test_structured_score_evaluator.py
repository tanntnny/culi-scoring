from __future__ import annotations

from pathlib import Path

from src.evaluators.structured_score_evaluator import StructuredScoreEvaluator, extract_score


class _DummyDataModule:
    def setup(self) -> None:
        return

    def train_dataloader(self):
        yield [([0.1, 0.2], 2.0), ([0.2, 0.3], 2.0)]


class _DummyModel:
    def forward(self, features):
        _ = features
        return 2.0


def test_extract_score_paths() -> None:
    assert extract_score('{"score": 3}') == 3
    assert extract_score("```json\n{\"score\": 2}\n```") == 2
    assert extract_score("score: 1") == 1
    assert extract_score("no score here") is None


def test_structured_score_evaluator_writes_outputs(tmp_path: Path) -> None:
    evaluator = StructuredScoreEvaluator(max_eval_batches=1)
    metrics = evaluator.evaluate(
        datamodule=_DummyDataModule(),
        model=_DummyModel(),
        metric_fn=lambda prediction, target: abs(prediction - target),
        run_dir=tmp_path,
    )

    assert metrics["total_samples"] == 2
    assert metrics["valid_samples"] == 2
    assert metrics["errors"] == 0
    assert (tmp_path / "evaluation_outputs.json").exists()
    assert (tmp_path / "confusion_matrix.json").exists()
    assert (tmp_path / "errors.json").exists()
    assert (tmp_path / "evaluation_metrics.json").exists()
    assert (tmp_path / "metrics_eval.json").exists()
