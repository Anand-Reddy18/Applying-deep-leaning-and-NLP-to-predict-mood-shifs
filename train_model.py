"""Evaluate a simple categorical transition baseline on the legacy CSV.

This deliberately replaces the old LSTM: integer label IDs are categories, not
continuous measurements, and the old trainer used a different sequence length
from its prediction script. No synthetic sleep/away labels are learned.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from mood_engine import hourly_majority, load_legacy_dataset


def main() -> None:
    raw = load_legacy_dataset("mood_dataset.csv")
    hourly = hourly_majority(raw)
    if len(hourly) < 12:
        raise SystemExit(
            f"Only {len(hourly)} valid emotion-hours are available. "
            "Collect more real observations before evaluating a forecast."
        )

    labels = hourly["emotion"].tolist()
    split = max(1, int(len(labels) * 0.8))
    if split >= len(labels):
        split = len(labels) - 1
    train_labels, test_labels = labels[:split], labels[split:]

    transitions = defaultdict(Counter)
    for previous, current in zip(train_labels, train_labels[1:]):
        transitions[previous][current] += 1
    global_counts = Counter(train_labels)

    predictions = []
    previous = train_labels[-1]
    for actual in test_labels:
        counts = transitions.get(previous, global_counts)
        prediction = counts.most_common(1)[0][0]
        predictions.append(prediction)
        previous = actual  # teacher-forced one-step-ahead evaluation

    encoder = LabelEncoder().fit(labels)
    metrics = {
        "valid_hourly_observations": len(hourly),
        "training_observations": len(train_labels),
        "test_observations": len(test_labels),
        "accuracy": accuracy_score(test_labels, predictions),
        "macro_f1": f1_score(test_labels, predictions, labels=encoder.classes_, average="macro", zero_division=0),
        "classification_report": classification_report(test_labels, predictions, labels=encoder.classes_, zero_division=0, output_dict=True),
        "note": "Small, user-specific dataset; chronological one-step evaluation is not proof of general predictive ability.",
    }
    Path("evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "classification_report"}, indent=2))
    print("\nDetailed report saved to evaluation.json")


if __name__ == "__main__":
    main()
