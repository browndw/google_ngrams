import numpy as np
import polars as pl


def is_sequence(x, tol=np.sqrt(np.finfo(float).eps), *args):
    if (np.any(np.isnan(x)) or
        np.any(np.isinf(x)) or len(x) <= 1 or
        np.diff(x[:2]) == 0):
        return False
    return np.diff(np.range(np.diff(x))).max() <= tol


def vnc_clust(time, values, distance_measure='sd'):
    if len(time) != len(values):
        raise ValueError("Your time and values vectors must be the same length.")

    input_values = np.array(values).copy()
    years = np.array(time).copy()

    data_collector = {}
    data_collector["0"] = input_values
    position_collector = {}
    position_collector[1] = 0
    overall_distance = 0
    number_of_steps = len(input_values) - 1

    for i in range(1, number_of_steps + 1):
        difference_checker = []
        unique_years = np.unique(years)

        for j in range(len(unique_years) - 1):
            first_name = unique_years[j]
            second_name = unique_years[j + 1]
            pooled_sample = input_values[np.isin(years, [first_name,
                                                         second_name])]

            if distance_measure == "sd":
                difference_checker.append(0 if np.sum(pooled_sample) == 0
                                          else np.std(pooled_sample, ddof=1))
            elif distance_measure == "cv":
                difference_checker.append(
                    0 if np.sum(pooled_sample) == 0
                    else np.std(pooled_sample, ddof=1) / np.mean(pooled_sample)
                    )

        pos_to_be_merged = np.argmin(difference_checker)
        distance = np.min(difference_checker)
        overall_distance += distance
        lower_name = unique_years[pos_to_be_merged]
        higher_name = unique_years[pos_to_be_merged + 1]

        matches = np.isin(years, [lower_name, higher_name])
        new_mean_age = round(np.mean(years[matches]), 4)
        position_collector[i + 1] = np.where(matches)[0] + 1
        years[matches] = new_mean_age
        data_collector[str(distance)] = input_values

    hc_build = pl.DataFrame({
        'start': [min(pos) if isinstance(pos, (list, np.ndarray)) else pos for pos in position_collector.values()],
        'end': [max(pos) if isinstance(pos, (list, np.ndarray)) else pos for pos in position_collector.values()]
    })

    idx = np.arange(len(hc_build))

    y = [np.where(
        hc_build['start'].to_numpy()[:i] == hc_build['start'].to_numpy()[i]
        )[0] for i in idx]
    z = [np.where(
        hc_build['end'].to_numpy()[:i] == hc_build['end'].to_numpy()[i]
        )[0] for i in idx]

    merge1 = [y[i].max().item() if len(y[i]) else np.nan for i in range(len(y))]
    merge2 = [z[i].max().item() if len(z[i]) else np.nan for i in range(len(z))]

    hc_build = (
        hc_build.with_columns([
            pl.Series('merge1',
                      [
                          min(m1, m2) if not np.isnan(m1) and
                          not np.isnan(m2) else np.nan for m1, m2 in zip(merge1, merge2)
                          ]),
            pl.Series('merge2',
                      [
                          max(m1, m2) for m1, m2 in zip(merge1, merge2)
                          ])
                      ])
    )

    hc_build = (
        hc_build.with_columns([
            pl.Series('merge1', [
                min(m1, m2) if not np.isnan(m1) and
                not np.isnan(m2) else np.nan for m1, m2 in zip(merge1, merge2)
                ]),
            pl.Series('merge2', [
                max(m1, m2) for m1, m2 in zip(merge1, merge2)
                ])
        ])
        )

    hc_build = (
        hc_build.with_columns([
            pl.when(
                pl.col('merge1').is_nan() &
                pl.col('merge2').is_nan()
                ).then(-pl.col('start')
                       ).otherwise(pl.col('merge1')).alias('merge1'),
            pl.when(
                pl.col('merge2')
                .is_nan()
                ).then(-pl.col('end')
                       ).otherwise(pl.col('merge2')).alias('merge2')
            ])
            )

    to_merge = [-np.setdiff1d(
        hc_build.select(
            pl.col('start', 'end')
            ).row(i),
        hc_build.select(
            pl.col('start', 'end')
            ).slice(1, i-1).to_numpy().flatten()
        ) for i in idx]

    to_merge = [-np.setdiff1d(
        hc_build.select(
            pl.col('start', 'end')
            ).row(i),
        hc_build.select(
            pl.col('start', 'end')
            ).slice(1, i-1).to_numpy().flatten()
        ) for i in idx]

    to_merge = [x[0].item() if len(x) > 0 else np.nan for x in to_merge]

    hc_build = (
        hc_build
        .with_columns([
            pl.when(pl.col('merge1').is_nan()
                    ).then(pl.Series(to_merge, strict=False)
                           ).otherwise(pl.col('merge1')).alias('merge1')
                           ])
                    )

    n = hc_build.height
    hc_build = (hc_build
                .with_columns(
                    pl.when(pl.col("merge1").lt(0))
                    .then(pl.col("merge1").mul(-1).sub(1))
                    .otherwise(pl.col('merge1').add(n)).alias('merge1')
                    )
                .with_columns(
                    pl.when(pl.col("merge2").lt(0))
                    .then(pl.col("merge2").mul(-1).sub(1))
                    .otherwise(pl.col('merge2').add(n)).alias('merge2')
                    )
                )

    hc_build = (
        hc_build
        .with_columns(distance=np.array(list(data_collector.keys())))
        .with_columns(pl.col("distance").cast(pl.Float64))
        .with_columns(pl.col("distance").cum_sum().alias("distance"))
        )

    size = np.array(
        [
            len(x) if isinstance(x, (list, np.ndarray))
            else 1 for x in position_collector.values()
         ])

    hc_build = (
        hc_build
        .with_columns(size=size)
        .with_columns(pl.col("size").cast(pl.Float64))
        )

    hc_build = hc_build.with_row_index().filter(pl.col("index") != 0)

    hc = hc_build.select("merge1", "merge2", "distance", "size").to_numpy()

    return hc
