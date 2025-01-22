# flake8: noqa

from importlib_resources import files as _files

sources = {
    "eng_all": _files("google_ngrams") / "data/googlebooks_eng_all_totalcounts_20120701.parquet",
    "gb_all": _files("google_ngrams") / "data/googlebooks_eng_gb_all_totalcounts_20120701.parquet",
    "us_all": _files("google_ngrams") / "data/googlebooks_eng_us_all_totalcounts_20120701.parquet",
}


def __dir__():
    return list(sources)

