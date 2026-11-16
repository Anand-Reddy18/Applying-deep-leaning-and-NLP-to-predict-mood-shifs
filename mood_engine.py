"""Shared mood-signal and history utilities.

Facial expressions and text sentiment are signals, not ground-truth mental states.
The forecast is an experimental estimate from self-reported history only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd

DB_PATH = Path("mood_history.db")
VALID_MOODS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
NON_MOOD_LABELS = {"away", "sleep", "unknown", "no_face", "none", ""}


def normalize_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower()
    return label if label in VALID_MOODS else None


def init_db(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS mood_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                mood TEXT NOT NULL,
                source TEXT NOT NULL,
                note TEXT DEFAULT ''
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_mood_timestamp ON mood_entries(timestamp)")


def add_entry(mood: str, source: str, note: str = "", db_path: Path = DB_PATH) -> None:
    normalized = normalize_label(mood)
    if normalized is None:
        raise ValueError(f"Unsupported mood label: {mood}")
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO mood_entries(timestamp, mood, source, note) VALUES (?, ?, ?, ?)",
            (datetime.now().astimezone().isoformat(timespec="seconds"), normalized, source, note[:500]),
        )


def read_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(
            "SELECT timestamp, mood, source, note FROM mood_entries ORDER BY timestamp",
            con,
        )


def next_mood_forecast(history: pd.DataFrame, min_entries: int = 10) -> dict:
    """First-order transition estimate using self-reported entries only."""
    if history.empty or "mood" not in history:
        return {"available": False, "reason": "No mood history has been saved yet."}

    own = history[history["source"] == "self_report"].copy()
    own["timestamp"] = pd.to_datetime(own["timestamp"], errors="coerce")
    own = own.dropna(subset=["timestamp"])
    own["mood"] = own["mood"].map(normalize_label)
    own = own.dropna(subset=["mood"]).sort_values("timestamp")

    if len(own) < min_entries:
        return {
            "available": False,
            "reason": f"Need at least {min_entries} self-reported entries; currently have {len(own)}.",
        }

    # Count transitions between consecutive self-reports. These are not assumed
    # to be equally spaced, so the result is explicitly a rough tendency.
    moods = own["mood"].tolist()
    transitions = defaultdict(Counter)
    for previous, current in zip(moods, moods[1:]):
        transitions[previous][current] += 1

    last = moods[-1]
    counts = transitions[last]
    if not counts:
        counts = Counter(moods)

    total = sum(counts.values())
    probabilities = {label: count / total for label, count in counts.items()}
    predicted = max(probabilities, key=probabilities.get)
    return {
        "available": True,
        "predicted": predicted,
        "probability": probabilities[predicted],
        "probabilities": probabilities,
        "last_mood": last,
        "entries_used": len(own),
        "caveat": "Experimental pattern estimate, not a reliable prediction of your actual future mood.",
    }


def load_legacy_dataset(path: str | Path = "mood_dataset.csv") -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    required = {"timestamp", "emotion"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["emotion"] = df["emotion"].map(normalize_label)
    df = df.dropna(subset=["timestamp", "emotion"]).sort_values("timestamp")
    return df


def hourly_majority(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observations to hourly bins; missing hours remain missing."""
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "emotion"])
    indexed = df.set_index("timestamp")
    result = indexed["emotion"].resample("1h").agg(
        lambda values: values.mode().iloc[0] if not values.mode().empty else None
    )
    result = result.dropna().reset_index()
    return result
