"""trisk_tfl — TRISK, the transition risk stress test, in Python (Theia Finance Labs).

Created with the support of AI tools.
"""
from .model import ENGINE_VERSION as __version__, R_COMMIT, R_VERSION, run_trisk_model

__all__ = ["run_trisk_model", "R_VERSION", "R_COMMIT", "__version__"]
