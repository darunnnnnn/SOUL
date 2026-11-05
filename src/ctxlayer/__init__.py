"""ctxlayer -- a knowledge base as a context layer.

Importing the package hardens the environment (torch-only HF backend, UTF-8
stdio) before any heavy dependency is touched. See ``ctxlayer._env``.
"""

from . import _env as _env  # noqa: F401  (import for side effects, first)

__version__ = "0.1.0"
