import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import dendrogram, fcluster


def _check_hierarchy(Z):
    n = Z.shape[0] + 1
    for i in range(0, n - 1):
        if Z[i, 0] >= n + i or Z[i, 1] >= n + i:
            return True
    return False


def _linkage_matrix(time_series,
                    frequency,
                    distance_measure='sd'):

    input_values = frequency.copy()
    years = time_series.copy()

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
            pooled_sample = input_values[np.isin(years,
                                                 [first_name,
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
        'start': [
            min(pos)
            if isinstance(pos, (list, np.ndarray))
            else pos for pos in position_collector.values()
            ],
        'end': [
            max(pos)
            if isinstance(pos, (list, np.ndarray))
            else pos for pos in position_collector.values()
            ]
    })

    idx = np.arange(len(hc_build))

    y = [np.where(
        hc_build['start'].to_numpy()[:i] == hc_build['start'].to_numpy()[i]
        )[0] for i in idx]
    z = [np.where(
        hc_build['end'].to_numpy()[:i] == hc_build['end'].to_numpy()[i]
        )[0] for i in idx]

    merge1 = [
        y[i].max().item() if len(y[i]) else np.nan for i in range(len(y))
        ]
    merge2 = [
        z[i].max().item() if len(z[i]) else np.nan for i in range(len(z))
        ]

    hc_build = (
        hc_build.with_columns([
            pl.Series('merge1',
                      [
                        min(m1, m2) if not np.isnan(m1) and
                        not np.isnan(m2)
                        else np.nan for m1, m2 in zip(merge1, merge2)
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
        .with_columns(
            pl.col("merge2").sub(pl.col("merge1")).alias("delta"))
            )
    hc_build = (
        hc_build
        .with_row_index()
        .with_columns(
            pl.when(
                pl.col("merge2").ge(pl.col("index").add(n - 1))
                ).then(pl.col("index").add(n - 1).sub(1).alias("merge2")
                       ).otherwise(pl.col("merge2"))
                )
                )
    hc_build = (
        hc_build
        .with_columns(
            pl.when(
                pl.col("merge1").gt(n - 1)
                ).then(pl.col("merge2").sub(pl.col("delta")).alias("merge1")
                       ).otherwise(pl.col("merge1"))
            )
            )
    hc_build = (
        hc_build
        .with_columns(distance=np.array(list(data_collector.keys())))
        .with_columns(pl.col("distance").cast(pl.Float64))
        .with_columns(pl.col("distance").cum_sum().alias("distance"))
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

    hc_build = hc_build.filter(pl.col("index") != 0)

    merge_cols = ["merge1", "merge2"]

    hc_build = (
        hc_build.with_columns(
            pl.when(
                pl.col("size").eq(2)
            )
            .then(pl.struct(
                merge1="merge2",
                merge2="merge1"
            ))
            .otherwise(pl.struct(merge_cols))
            .struct.field(merge_cols)
        )
    )

    hc = hc_build.select("merge1", "merge2", "distance", "size").to_numpy()
    return hc


class TimeSeries:

    def __init__(self,
                 time_series: pl.DataFrame,
                 time_col: str,
                 values_col: str):

        time = time_series.get_column(time_col, default=None)
        values = time_series.get_column(values_col, default=None)

        if time is None:
            raise ValueError("""
                Invalid column.
                Check name. Couldn't find column in DataFrame.
                    """)
        if values is None:
            raise ValueError("""
                Invalid column.
                Check name. Couldn't find column in DataFrame.
                """)
        if not isinstance(values.dtype, (pl.Float64, pl.Float32)):
            raise ValueError("""
                Invalid DataFrame.
                Expected a column of normalized frequencies.
                """)
        if len(time) != len(values):
            raise ValueError("""
                Your time and values vectors must be the same length.
                """)

        time_series = time_series.sort(time)
        self.time_intervals = time_series.get_column(time_col).to_numpy()
        self.frequencies = time_series.get_column(values_col).to_numpy()
        self.Z_sd = _linkage_matrix(time_series=self.time_intervals,
                                    frequency=self.frequencies)
        self.Z_cv = _linkage_matrix(time_series=self.time_intervals,
                                    frequency=self.frequencies,
                                    distance_measure='cv')
        self.distances_sd = np.array([self.Z_sd[i][2].item()
                                      for i in range(len(self.Z_sd))])
        self.distances_cv = np.array([self.Z_cv[i][2].item()
                                      for i in range(len(self.Z_cv))])

        self.clusters = None
        self.distance_threshold = None

    def timeviz_vnc(self,
                    width=6,
                    height=4,
                    dpi=150,
                    n_periods=1,
                    font_size=10,
                    distance="sd",
                    orientation="horizontal",
                    rotate_labels=False,
                    cut_line=False,
                    periodize=False) -> Figure:

        dist_types = ['sd', 'cv']
        if distance not in dist_types:
            distance = "sd"

        labels_ = self.time_intervals

        if distance == "cv":
            Z = self.Z_cv
        else:
            Z = self.Z_sd

        if n_periods > len(Z):
            n_periods = 1

        if rotate_labels is True:
            rotation = 90
        else:
            rotation = 0

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        # assign leaves to clusters
        clusters_assignment = fcluster(Z, n_periods, criterion='maxclust')
        clusters_assignment = {
            str(labels_[i]): clusters_assignment[i].item()
            for i in range(len(clusters_assignment))
            }
        clusters = {}
        for key, value in clusters_assignment.items():
            if value not in clusters:
                clusters[value] = []
            clusters[value].append(key)
        self.clusters = {
            f"Cluster {i + 1}": list(clusters.values())[i]
            for i in range(len(list(clusters.values())))
            }

        # set distance threshold to mean between heights
        if n_periods > 1:
            dist_ = [x[2].item() for x in Z]
            dist_thr = np.mean(
                [dist_[len(dist_)-n_periods+1], dist_[len(dist_)-n_periods]]
                )

        # Plot the corresponding dendrogram
        if orientation == "horizontal" and periodize is not True:
            dendrogram(Z,
                       ax=ax,
                       labels=labels_,
                       distance_sort="descending",
                       leaf_font_size=font_size,
                       leaf_rotation=rotation,
                       color_threshold=0,
                       above_threshold_color='k')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel(f'Distance (in summed {distance})')
            plt.setp(ax.collections, linewidth=.5)

            if cut_line and n_periods > 1:
                ax.axhline(y=dist_thr,
                           color='r',
                           alpha=0.7,
                           linestyle='--',
                           linewidth=.5)

        if orientation == "horizontal" and periodize is True:
            dendrogram(Z,
                       ax=ax,
                       labels=labels_,
                       distance_sort="descending",
                       leaf_font_size=font_size,
                       leaf_rotation=rotation,
                       truncate_mode='lastp',
                       p=n_periods,
                       get_leaves=True,
                       show_leaf_counts=True,
                       show_contracted=True,
                       color_threshold=0,
                       above_threshold_color='k')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel(f'Distance (in summed {distance})')
            plt.setp(ax.collections, linewidth=.5)

        if orientation == "vertical" and periodize is not True:
            dendrogram(Z,
                       ax=ax,
                       labels=labels_,
                       leaf_font_size=font_size,
                       orientation="right",
                       color_threshold=0,
                       above_threshold_color='k')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel('Distance (in summed sd)')
            plt.setp(ax.collections, linewidth=.5)

            if cut_line and n_periods > 1:
                ax.axvline(x=dist_thr,
                           color='r',
                           alpha=0.7,
                           linestyle='--',
                           linewidth=.5)

        if orientation == "vertical" and periodize is True:
            dendrogram(Z,
                       ax=ax,
                       labels=labels_,
                       leaf_font_size=font_size,
                       orientation="right",
                       truncate_mode='lastp',
                       p=n_periods,
                       get_leaves=True,
                       show_leaf_counts=True,
                       show_contracted=True,
                       color_threshold=0,
                       above_threshold_color='k')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_ylabel(f'Distance (in summed {distance})')
            plt.setp(ax.collections, linewidth=.5)

        return fig
