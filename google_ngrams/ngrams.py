
import re
import polars as pl
from textwrap import dedent
from data import sources


def google_ngram(word_forms,
                 variety="eng",
                 by="decade"):
    variety_types = ["eng", "gb", "us", "fiction"]
    if variety not in variety_types:
        raise ValueError("""variety_types
                         Invalid variety type. Expected one of: %s
                         """ % variety_types)
    by_types = ["year", "decade"]
    if by not in by_types:
        raise ValueError("""variety_types
                         Invalid by type. Expected one of: %s
                         """ % by_types)
    word_forms = [re.sub(r'([a-zA-Z0-9])-([a-zA-Z0-9])',
                         r'\1 - \2', wf) for wf in word_forms]
    word_forms = [wf.strip() for wf in word_forms]
    n = [len(re.findall(r'\S+', wf)) for wf in word_forms]
    n = list(set(n))

    if len(n) > 1:
        raise ValueError("""Check spelling.
                         Word forms should be lemmas of the same word
                         (e.g. 'teenager' and 'teenagers'
                         or 'walk', 'walks' and 'walked'
                         """)
    if n[0] > 5:
        raise ValueError("""Ngrams can be a maximum of 5 tokens.
                         Hyphenated words are split and include the hyphen,
                         so 'x-ray' would count as 3 tokens.
                         """)

    gram = [wf[:2] if n[0] > 1 else wf[:1] for wf in word_forms]
    gram = list(set([g.lower() for g in gram]))

    if len(gram) > 1:
        raise ValueError("""Check spelling.
                         Word forms should be lemmas of the same word
                         (e.g. 'teenager' and 'teenagers'
                         or 'walk', 'walks' and 'walked'
                         """)

    if re.match(r'^[a-z][^a-z]', gram[0]):
        gram[0] = re.sub(r'[^a-z]', '_', gram[0])
    if re.match(r'^[0-9]', gram[0]):
        gram[0] = gram[0][:1]
    if re.match(r'^[\W]', gram[0]):
        gram[0] = "punctuation"

    if any(re.match(r'^[ßæðøłœıƒþȥəħŋªºɣđĳɔȝⅰʊʌʔɛȡɋⅱʃɇɑⅲ]', g) for g in gram):
        gram[0] = "other"

    gram[0] = gram[0].encode('latin-1', 'replace').decode('latin-1')

    if variety == "eng":
        repo = f"http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-{n[0]}gram-20120701-{gram[0]}.gz"  # noqa: E501
    else:
        repo = f"http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-{variety}-all-{n[0]}gram-20120701-{gram[0]}.gz"  # noqa: E501

    print(dedent(
        """
        Accessing repository. For larger ones
        (e.g., ngrams containing 2 or more words).
        This may take a few minutes. A progress bar should appear shortly...
        """
    ))

    word_forms = [re.sub(
        r'(\.|\?|\$|\^|\)|\(|\}|\{|\]|\[|\*)',
        r'\\\1', wf
        ) for wf in word_forms]

    grep_words = "|".join([f"^{wf}$" for wf in word_forms])

    # Read the data from the google repository and format
    df = pl.scan_csv(repo, separator='\t', has_header=False)
    filtered_df = df.filter(
        pl.col("column_1").str.contains(r"(?i)" + grep_words)
        )
    all_grams = filtered_df.collect()

    all_grams = (
        all_grams
        .rename(
            {"column_1": "Token",
             "column_2": "Year",
             "column_3": "AF"}
             )
        ).drop("column_4")

    # read totals
    if variety == "eng":
        f_path = sources.get("eng_all")
    elif variety == "gb":
        f_path = sources.get("gb_all")
    elif variety == "us":
        f_path = sources.get("us_all")

    total_counts = pl.read_parquet(f_path)
    total_counts = total_counts.cast({"Year": pl.UInt32,
                                      "Total": pl.UInt64,
                                      "Pages": pl.UInt64,
                                      "Volumes": pl.UInt64})
    # format totals, fill missing data, and sum
    total_counts = total_counts.cast({
        "Year": pl.UInt32,
        "Total": pl.UInt64,
        "Pages": pl.UInt64,
        "Volumes": pl.UInt64
        })

    total_counts = (
        total_counts
        .with_columns(
            pl.col("Year")
            .cast(pl.String).str.to_datetime("%Y")
            )
        .sort("Year")
        .upsample(time_column="Year", every="1y")
        .with_columns(
            pl.col(["Total", "Pages", "Volumes"])
            .fill_null(strategy="zero")
            )
            )
    total_counts = (
        total_counts
        .group_by_dynamic(
            "Year", every="1y"
        ).agg(pl.col("Total").sum())
    )

    # sum token totals, convert to datetime and fill in missing years
    sum_tokens = all_grams.group_by("Year", maintain_order=True).sum()
    sum_tokens = (
        sum_tokens
        .with_columns(
            pl.col("Year")
            .cast(pl.String).str.to_datetime("%Y")
            )
        .sort("Year")
        .upsample(time_column="Year", every="1y")
        .with_columns(
                pl.col("AF")
                .fill_null(strategy="zero")
                )
        ).drop("Token")
    # join with totals
    sum_tokens = sum_tokens.join(total_counts, on="Year")

    if by == "decade":
        sum_tokens = (
            sum_tokens
            .group_by_dynamic("Year", every="10y")
            .agg(pl.col(["AF", "Total"]).sum())
        )
    # normalize RF per million tokens
    sum_tokens = (
        sum_tokens
        .with_columns(
            RF=pl.col("AF").truediv("Total").mul(1000000)
            )
        .with_columns(
            pl.col("RF").fill_nan(0)
            )
    )
    sum_tokens.insert_column(1, (pl.lit(word_forms)).alias("Token"))

    if by == "decade":
        sum_tokens = sum_tokens.rename({"Year": "Decade"})

    return sum_tokens
