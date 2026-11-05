"""Environment hardening, imported before anything touches ``transformers``.

HuggingFace ``transformers`` probes for TensorFlow/Flax at import time and will
happily crash the whole process if a *broken but installed* TF is present (a
very common state on shared machines -- e.g. a protobuf version mismatch). We
are a torch-only codebase, so we pin the backend explicitly rather than relying
on the probe. Doing this inside the package, not in a shell script, is what
makes ``import ctxlayer`` safe for anyone reproducing the evaluation.

Also fixes stdout to UTF-8: the corpus is Wikipedia text full of non-Latin-1
characters, and the Windows console defaults to cp1252, which turns an
otherwise-successful run into a UnicodeEncodeError at print time.
"""

from __future__ import annotations

import os
import sys

# Must precede `import transformers`.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Determinism: cap intra-op threading jitter is not needed for correctness, but
# an unbounded thread count on a laptop makes latency measurements noisy.
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, (os.cpu_count() or 4))))


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


ensure_utf8_stdio()


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/torch. Called by the eval runner before every system."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU matmul is already stable
    except Exception:
        pass
