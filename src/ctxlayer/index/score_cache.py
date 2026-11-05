"""Persistent cross-encoder score cache.

The cross-encoder is the dominant cost of the whole evaluation. Our system
issues round 0 plus up to ``max_rounds x beam_entities`` bridge queries per
question, so a 500-question run reranks on the order of 10^5 (query, passage)
pairs -- minutes to hours on CPU.

Crucially, those pairs **repeat heavily across the runs we need**:

* every system that reranks sees the same round-0 query (the question) over a
  largely overlapping candidate list;
* an ablation changes one config field, so most of its bridge queries are
  identical to the full system's;
* re-running the evaluation to regenerate the report (Hard Requirement #4)
  repeats everything.

A content-addressed cache therefore turns the ablation suite from "N x the cost
of one run" into roughly "one run plus the deltas", which is what makes a
one-command full reproduction practical.

Correctness: the key covers the model name, the truncation length and the exact
(query, passage text) pair, so a changed model or ``max_length`` misses the
cache rather than silently returning scores from a different model. Scores are
pure functions of those inputs under ``inference_mode``, so caching cannot
change any result -- it is a speed optimisation only, and
``tests/test_cache.py`` asserts cached and uncached scores agree.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

__all__ = ["ScoreCache"]


class ScoreCache:
    """SQLite-backed (key -> float) store. Thread-safe, process-safe."""

    def __init__(self, path: Path | None, model: str, max_length: int) -> None:
        self.model = model
        self.max_length = max_length
        self.enabled = path is not None
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._pending: dict[str, float] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), check_same_thread=False)
            # WAL + NORMAL sync: the cache is a derived artefact, so trading a
            # crash-window for throughput is the right call.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS scores (k TEXT PRIMARY KEY, v REAL)"
            )
            self._conn.commit()

    def key(self, query: str, passage_text: str) -> str:
        return hashlib.sha1(
            "\x1f".join(
                [self.model, str(self.max_length), query, passage_text]
            ).encode("utf-8")
        ).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        if self._conn is None or not keys:
            return {}
        out: dict[str, float] = {}
        with self._lock:
            # Chunked to stay under SQLite's variable limit (999 by default).
            for i in range(0, len(keys), 800):
                chunk = keys[i : i + 800]
                placeholders = ",".join("?" * len(chunk))
                for k, v in self._conn.execute(
                    f"SELECT k, v FROM scores WHERE k IN ({placeholders})", chunk
                ):
                    out[k] = float(v)
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, items: dict[str, float]) -> None:
        if self._conn is None or not items:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO scores (k, v) VALUES (?, ?)",
                list(items.items()),
            )
            self._conn.commit()

    def stats(self) -> dict[str, float]:
        total = self.hits + self.misses
        return {
            "cache_hits": float(self.hits),
            "cache_misses": float(self.misses),
            "cache_hit_rate": self.hits / total if total else 0.0,
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
