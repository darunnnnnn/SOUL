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
# Use all physical cores for the CPU matmuls that dominate encoding/reranking.
# Measured on this 16-core box: bge-small corpus encoding runs at 14.6 psg/s
# with the 8-thread default and 18.3 psg/s with 16 -- a ~25% saving on a cost
# that is paid on every index rebuild.
_THREADS = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS", _THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _THREADS)


def configure_threads() -> None:
    """Align torch's intra-op pool with OMP_NUM_THREADS.

    torch reads its default thread count at import time and does not always
    follow OMP_NUM_THREADS, so it is set explicitly.
    """
    try:
        import torch

        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    except Exception:
        pass


def ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


ensure_utf8_stdio()
configure_threads()


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
