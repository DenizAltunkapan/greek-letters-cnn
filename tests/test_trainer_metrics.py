import pytest
import torch

from trainer import format_per_class_report, per_class_metrics, top_confusions


def test_per_class_metrics_handles_empty_support_and_zero_predictions():
    confusion = torch.tensor(
        [
            [2, 1, 0],
            [0, 0, 0],
            [1, 0, 3],
        ]
    )

    rows = per_class_metrics(confusion, ["Alpha", "Beta", "Gamma"])

    assert rows[0]["support"] == 3
    assert rows[0]["correct"] == 2
    assert rows[0]["recall"] == pytest.approx(2 / 3)
    assert rows[0]["precision"] == pytest.approx(2 / 3)
    assert rows[1] == {
        "class": "Beta",
        "support": 0,
        "correct": 0,
        "recall": 0.0,
        "precision": 0.0,
        "f1": 0.0,
    }
    assert rows[2]["recall"] == pytest.approx(3 / 4)
    assert rows[2]["precision"] == pytest.approx(1.0)


def test_top_confusions_excludes_correct_predictions_and_sorts_by_count():
    confusion = torch.tensor(
        [
            [9, 3, 1],
            [2, 8, 4],
            [0, 5, 7],
        ]
    )

    pairs = top_confusions(confusion, ["Alpha", "Beta", "Gamma"], top_k=3)

    assert pairs == [
        ("Gamma", "Beta", 5),
        ("Beta", "Gamma", 4),
        ("Alpha", "Beta", 3),
    ]


def test_format_per_class_report_orders_worst_recall_first():
    rows = [
        {"class": "Alpha", "support": 10, "correct": 9, "recall": 0.9, "precision": 1.0, "f1": 0.947},
        {"class": "Beta", "support": 10, "correct": 5, "recall": 0.5, "precision": 0.7, "f1": 0.583},
    ]

    report = format_per_class_report(rows)

    assert report.splitlines()[3].startswith("Beta")
    assert "macro avg recall: 0.700" in report
    assert "macro avg f1: 0.765" in report
