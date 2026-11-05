"""Dense bi-encoder index (bge-small-en-v1.5), exact search.

Model choice
------------
``BAAI/bge-small-en-v1.5``: 33M params / 384-dim / 133 MB. On the MTEB
retrieval suite it is within ~1-2 nDCG@10 of ``bge-base`` at a quarter of the
size, and it runs at usable speed on CPU -- which keeps the whole evaluation
reproducible on a laptop with no GPU and no API key. Since the dense retriever
is held *identical* across our system and every retrieval baseline, its
absolute strength does not bias any comparison; using a bigger encoder would
move all rows of the results table in the same direction.

Exact rather than approximate search
------------------------------------
The knowledge base is ~7k-21k passages. A brute-force normalised matmul is
~8 MFLOPs per query (<1 ms) and is *exact*, so no ANN recall ceiling is
silently folded into our retrieval numbers. Adding FAISS/HNSW here would make
results depend on index hyperparameters while measurably losing recall. ANN is
the right call at 10^7 passages; at 10^4 it is strictly worse.

Pooling
-------
bge-v1.5 is trained with **CLS pooling** and an asymmetric query instruction.
Using mean pooling (the common copy-paste error) costs several points of nDCG.
The instruction prefix is applied to queries only, never to passages.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..config import DenseConfig
from ..schema import Passage

__all__ = ["DenseIndex", "resolve_device"]


def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch always present here
        pass
    return "cpu"


class _Encoder:
    """Lazily-loaded HF encoder with CLS pooling and L2 normalisation."""

    _cache: dict[tuple[str, str], "_Encoder"] = {}

    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        # Inference-only: disable grad globally for this module's forwards.
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def get(cls, model_name: str, device: str) -> "_Encoder":
        key = (model_name, device)
        if key not in cls._cache:
            cls._cache[key] = cls(model_name, device)
        return cls._cache[key]

    def encode(
        self,
        texts: Sequence[str],
        max_length: int,
        batch_size: int,
        normalize: bool,
        progress: str | None = None,
    ) -> np.ndarray:
        torch = self.torch
        n = len(texts)
        if n == 0:
            return np.zeros((0, 384), dtype=np.float32)

        # Length-sorted batching. With dynamic padding, a batch costs the
        # *longest* sequence in it, so mixing an 8-token title with a 300-token
        # paragraph wastes most of the compute. Grouping similar lengths cut
        # corpus encoding time ~2.5x on CPU, and it also *reduces* numerical
        # noise by padding less. Results are restored to the input order.
        order = sorted(range(n), key=lambda i: len(texts[i]))
        vecs: list[np.ndarray | None] = [None] * n
        done = 0
        with torch.inference_mode():
            for i in range(0, n, batch_size):
                idx = order[i : i + batch_size]
                enc = self.tok(
                    [texts[j] for j in idx],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                vec = hidden[:, 0]  # CLS pooling -- required for bge-v1.5
                if normalize:
                    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                arr = vec.float().cpu().numpy()
                for row, j in enumerate(idx):
                    vecs[j] = arr[row]
                done += len(idx)
                if progress and (i // batch_size) % 10 == 0:
                    print(f"\r  {progress}: {done}/{n}", end="", flush=True)
        if progress:
            print(f"\r  {progress}: {n}/{n}", flush=True)
        return np.stack(vecs).astype(np.float32)  # type: ignore[arg-type]


class DenseIndex:
    # Upper bound on the in-memory query-embedding cache (~1.5 MB at 384 dims).
    _QCACHE_MAX = 20000

    def __init__(self, cfg: DenseConfig) -> None:
        self.cfg = cfg
        self.passage_ids: list[str] = []
        self._pid_to_row: dict[str, int] = {}
        self.emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.device = resolve_device(cfg.device)
        self._qcache: dict[str, np.ndarray] = {}

    # ---------------------------------------------------------------- build
    def build(
        self,
        passages: Sequence[Passage],
        cache_dir: Path | None = None,
        verbose: bool = True,
    ) -> "DenseIndex":
        self.passage_ids = [p.passage_id for p in passages]
        self._pid_to_row = {pid: i for i, pid in enumerate(self.passage_ids)}

        # Cache key covers everything that can change an embedding: the model,
        # truncation length, normalisation, the passage-side prefix, and the
        # exact ordered content being encoded. A config typo therefore produces
        # a cache miss instead of silently reusing stale vectors.
        key = hashlib.sha256(
            "\x1f".join(
                [
                    self.cfg.model,
                    str(self.cfg.max_length),
                    str(self.cfg.normalize),
                    self.cfg.passage_prefix,
                    *self.passage_ids,
                ]
            ).encode()
        ).hexdigest()[:20]

        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"dense_{key}.npy"
            if cache_path.exists():
                self.emb = np.load(cache_path)
                if self.emb.shape[0] == len(self.passage_ids):
                    if verbose:
                        print(f"  dense: loaded cached embeddings {self.emb.shape}")
                    return self
                self.emb = np.zeros((0, 0), dtype=np.float32)  # stale; re-encode

        enc = _Encoder.get(self.cfg.model, self.device)
        texts = [self.cfg.passage_prefix + p.with_title() for p in passages]
        self.emb = enc.encode(
            texts,
            self.cfg.max_length,
            self.cfg.batch_size,
            self.cfg.normalize,
            progress=f"dense encode ({self.device})" if verbose else None,
        )
        if cache_path is not None:
            np.save(cache_path, self.emb)
        return self

    # --------------------------------------------------------------- search
    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Encode queries, memoising by exact prefixed string.

        The iterative retriever issues ~13 queries per question, and bridge
        queries recur heavily across systems, ablations and repeated runs
        ("<entity> <relation terms>" is a function of the question and the
        entity, both of which repeat). Encoding is ~10 ms, so this is ~65 s per
        500-question system and ~20 min across an ablation sweep, spent
        recomputing identical vectors.

        Exact-string keyed, so it cannot change a result. Bounded to keep
        memory flat on long runs; eviction is simple insertion-order trimming,
        which is adequate because queries within a run cluster in time.
        """
        enc = _Encoder.get(self.cfg.model, self.device)
        keys = [self.cfg.query_prefix + q for q in queries]
        unique = list(dict.fromkeys(keys))
        resolved: dict[str, np.ndarray] = {
            k: self._qcache[k] for k in unique if k in self._qcache
        }
        missing = [k for k in unique if k not in resolved]
        if missing:
            vecs = enc.encode(
                missing,
                self.cfg.max_length,
                self.cfg.batch_size,
                self.cfg.normalize,
            )
            for k, v in zip(missing, vecs):
                resolved[k] = v
                self._qcache[k] = v

        # Build the result from `resolved`, never from the cache: eviction
        # below must not be able to remove an entry this call still needs.
        # (Reading straight from the cache raised KeyError once the request
        # itself was larger than the cache bound.)
        out = np.stack([resolved[k] for k in keys])

        if len(self._qcache) > self._QCACHE_MAX:
            # Evict oldest-first, but never the keys just requested.
            protected = set(unique)
            for k in list(self._qcache):
                if len(self._qcache) <= self._QCACHE_MAX:
                    break
                if k not in protected:
                    del self._qcache[k]
        return out

    def score_all(self, query: str) -> np.ndarray:
        if self.emb.size == 0:
            return np.zeros(len(self.passage_ids), dtype=np.float32)
        qv = self.encode_queries([query])[0]
        return self.emb @ qv

    def search(
        self,
        query: str,
        top_k: int = 100,
        restrict_to: Iterable[str] | None = None,
        _scores: np.ndarray | None = None,
    ) -> list[tuple[str, float]]:
        scores = self.score_all(query) if _scores is None else _scores
        if restrict_to is not None:
            rows = np.fromiter(
                (self._pid_to_row[p] for p in restrict_to if p in self._pid_to_row),
                dtype=np.int64,
            )
            if rows.size == 0:
                return []
            sub = scores[rows]
            k = min(top_k, rows.size)
            sel = np.argpartition(-sub, k - 1)[:k]
            sel = sel[np.argsort(-sub[sel], kind="stable")]
            return [(self.passage_ids[int(rows[i])], float(sub[i])) for i in sel]
        k = min(top_k, len(self.passage_ids))
        if k == 0:
            return []
        sel = np.argpartition(-scores, k - 1)[:k]
        sel = sel[np.argsort(-scores[sel], kind="stable")]
        return [(self.passage_ids[int(i)], float(scores[i])) for i in sel]

    def vector_for(self, passage_id: str) -> np.ndarray | None:
        row = self._pid_to_row.get(passage_id)
        return None if row is None else self.emb[row]

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            emb=self.emb,
            passage_ids=np.array(self.passage_ids, dtype=object),
            cfg=np.array([repr(self.cfg.to_dict())], dtype=object),
        )

    @classmethod
    def load(cls, path: Path, cfg: DenseConfig) -> "DenseIndex":
        d = np.load(path, allow_pickle=True)
        idx = cls(cfg)
        idx.emb = d["emb"]
        idx.passage_ids = list(d["passage_ids"])
        idx._pid_to_row = {p: i for i, p in enumerate(idx.passage_ids)}
        return idx
