# flake8: noqa

# Set version ----
from importlib_metadata import version as _v

__version__ = _v("google_ngrams")

del _v

# Imports ----

from .ngrams import google_ngram

from .vnc import VNCAnalyzer

__all__ = ['google_ngram', 'VNCAnalyzer']