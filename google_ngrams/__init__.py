# flake8: noqa

# Set version ----
from importlib.metadata import version as _v

__version__ = _v("google_ngrams")

del _v

# Imports ----

from .ngrams import google_ngram

from .vnc import TimeSeries

__all__ = ['google_ngram', 'TimeSeries']