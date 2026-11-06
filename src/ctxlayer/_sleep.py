"""Keep the machine awake for the duration of a long run.

Why this is here
----------------
An overnight 500-question evaluation was charged 6.5 hours of wall time to a
single question because the laptop suspended mid-run. Nothing was wrong with
the computation -- median cost was 3.2 s/question -- but:

* ``time.time()`` is wall-clock, so the suspend landed inside one question's
  measured latency and corrupted that record;
* the progress ETA, computed as ``elapsed / completed``, inherited the suspend
  and projected 18 hours of remaining work against a true ~20 minutes, which is
  exactly the kind of signal that makes someone kill a healthy run.

So a long run now asks the OS to stay awake while it works. This is
*process-scoped*: the request is released when the process exits, and no system
power setting is modified, so it cannot leave a machine configured differently
from how it was found.

On Windows this is ``SetThreadExecutionState``; on macOS ``caffeinate``; on
Linux there is no portable equivalent we can rely on, so the call degrades to a
no-op rather than failing a run.

Implementation note: each platform branch must ``yield`` **exactly once**, and
the yield must not sit inside an ``except`` that could catch an exception
thrown back in by the ``with`` body. A first version wrapped the yield in a
broad ``try/except`` and so attempted a second yield while unwinding, raising
"generator didn't stop after throw()" and masking the real error underneath.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from typing import Iterator

__all__ = ["keep_awake"]

# SetThreadExecutionState flags (winbase.h)
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
# Deliberately NOT ES_DISPLAY_REQUIRED: the screen may sleep, the machine may
# not. Keeping a display lit all night is not ours to decide.


@contextlib.contextmanager
def keep_awake(reason: str = "ctxlayer evaluation") -> Iterator[bool]:
    """Best-effort sleep inhibition for the enclosing block.

    Yields True only if inhibition was actually established, so a caller can
    say so rather than imply a guarantee it does not have.
    """
    release = None
    established = False

    if sys.platform.startswith("win"):
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            established = bool(
                kernel32.SetThreadExecutionState(
                    _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
                )
            )
            if established:
                def release() -> None:  # noqa: E306
                    kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        except Exception:
            established = False

    elif sys.platform == "darwin":
        try:
            proc = subprocess.Popen(
                ["caffeinate", "-i", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            established = True

            def release() -> None:  # noqa: E306
                if proc.poll() is None:
                    proc.terminate()
        except Exception:
            established = False

    # Single yield on every path, outside any except that could swallow an
    # exception thrown back in from the body.
    try:
        yield established
    finally:
        if release is not None:
            with contextlib.suppress(Exception):
                release()
