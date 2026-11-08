"""SQLite persistence for candidate results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from shapehunter.core import Candidate


def store_candidates(path: str | Path, run_id: str, candidates: list[Candidate]) -> None:
    """Atomically replace one run's candidate list in a SQLite database."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS results (run_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (run_id, ordinal))")
        db.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        db.executemany("INSERT INTO results VALUES (?, ?, ?)", [(run_id, i, json.dumps(c.to_dict(), allow_nan=False)) for i, c in enumerate(candidates)])


def load_candidates(path: str | Path, run_id: str) -> list[Candidate]:
    """Read candidates for a run in their original order."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS results (run_id TEXT NOT NULL, ordinal INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (run_id, ordinal))")
        rows = db.execute("SELECT payload FROM results WHERE run_id = ? ORDER BY ordinal", (run_id,)).fetchall()
    return [Candidate.from_dict(json.loads(row[0])) for row in rows]
