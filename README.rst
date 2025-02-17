
google_ngrams: Fetch and analyze Google Ngram data for specified word forms.
=======================================================================================================
|pypi| |pypi_downloads|

This package has functions for processing `Google’s Ngram repositories <http://storage.googleapis.com/books/ngrams/books/datasetsv2.html>`_ without having to download them locally. These repositories vary in their size, but the larger ones (like th one for the letter *s* or common bigrams) contain multiple gigabytes.

The main function uses `scan_csv from the polars <https://docs.pola.rs/api/python/dev/reference/api/polars.scan_csv.html>`_ package to reduce memory load. Still, depending on the specific word forms being searched, loading and processing the data tables can sometimes take a few minutes if they are large.

vnc
---

To analyze the returned data, the package als contains functions based on the work of Gries and Hilpert (2012) for `Variability-Based Neighbor Clustering <https://www.oxfordhandbooks.com/view/10.1093/oxfordhb/9780199922765.001.0001/oxfordhb-9780199922765-e-14>`_.

The idea is to use hierarchical clustering to aid "bottom up  periodization of language change. The python functions are built on `their original R code <http://global.oup.com/us/companion.websites/fdscontent/uscompanion/us/static/companion.websites/nevalainen/Gries-Hilpert_web_final/vnc.individual.html>`_.

Distances, therefore, are calculated in sums of standard deviations and coefficients of variation, according to their stated method.

Dendrograms are plotted using matplotlib, following the scipy conventions for formatting coordinates. However, the package has customized functions for maintaining the plotting order of the leaves according the requirements of the method.

The package also has an implementation of `scipy's truncate_mode <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html/>`_ that consolidates leaves under a specified number of time periods (or clusters) while also maintaining the leaf order to facilitate the reading and interpretation of large dendrograms.


Installation
------------

You can install the released version of google_ngrams from `PyPI <https://pypi.org/project/google_ngrams/>`_:

.. code-block:: install-google_ngrams

    pip install google_ngrams


Usage
-----

To use the google_ngrams package, import :code:`google_ngram` to fetch data and :code:`TimeSeries` for analysis.

.. code-block:: import

    from google_ngrams import google_ngram, TimeSeries 


Fetching n-gram data
^^^^^^^^^^^^^^^^^^^^

The :code:`google_ngram` function supports different varieties of English (e.g., British, American) and allows aggregation by year or decade. Word forms (even a single word form) must be formatted as a list:

The following would return counts for the word *x-ray* in US English by year:

.. code-block:: by_year

    xray_year = google_ngram(word_forms = ["x-ray"], variety = "us", by = "year")

Alternatively, the following would return counts of the combined forms *xray* and *xrays* in British English by decade:

.. code-block:: by_decade

    xray_decade = google_ngram(word_forms = ["x-ray", "x-rays"], variety = "gb", by = "decade")

The function returns a polars DataFrame with either a time interval column (either :code:`Year` or :code:`Decade`) and columns for :code:`Token`, :code:`AF` (absolute frequency) and :code:`RF` (relative frequency).

The returned DataFrame, then, can be manipulated using the polars API:

.. code-block:: filtering

    import polars as pl
    
    xray_filtered = xray_decade.filter(pl.col("Decade") >= 1900)


Analyzing time series data
^^^^^^^^^^^^^^^^^^^^^^^^^^

To analyze the data, use :code:`TimeSeries`, specifying a column of time intervals and a column of relative frequencies:

.. code-block:: time_series

    xray_ts = TimeSeries(xray_filtered, time_col="Decade", values_col="RF")
    
VNC dendrograms can then be plotted with a variety of options:

.. code-block:: dendrogram

    xray_ts.timeviz_vnc()

For additional information, consult the `documentation <https://browndw.github.io/google_ngrams/>`_.


License
-------

Code licensed under `Apache License 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`_.
See `LICENSE <https://github.com/browndw/docuscospacy/blob/master/LICENSE>`_ file.

.. |pypi| image:: https://badge.fury.io/py/google_ngrams.svg
    :target: https://badge.fury.io/py/pybiber
    :alt: PyPI Version

.. |pypi_downloads| image:: https://img.shields.io/pypi/dm/google_ngrams
    :target: https://pypi.org/project/google_ngrams/
    :alt: Downloads from PyPI

