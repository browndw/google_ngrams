import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.cluster import hierarchy as sch


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
                max(m1, m2) if not np.isnan(m1)
                else m2 for m1, m2 in zip(merge1, merge2)
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

    hc_build = hc_build.with_row_index()
    n = hc_build.height

    hc_build = (hc_build
                .with_columns(
                    pl.when(pl.col("merge1").lt(0))
                    .then(pl.col("merge1").mul(-1).sub(1))
                    .otherwise(pl.col('merge1').add(n-1)).alias('merge1')
                    )
                .with_columns(
                    pl.when(pl.col("merge2").lt(0))
                    .then(pl.col("merge2").mul(-1).sub(1))
                    .otherwise(pl.col('merge2').add(n-1)).alias('merge2')
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

    hc = hc_build.select("merge1", "merge2", "distance", "size").to_numpy()
    return hc


def _find_first_previous_match(arr, value):
    """Finds the index of the first previous match in an array."""
    indices = np.where(arr == value)[0]
    if len(indices) > 0:
        return indices[0].item()
    else:
        np.nan


def _update_centers(df, left_matches, right_matches):
    centers = (
        df.sort("dcoord_2", descending=False)
        .with_columns(centers=pl.mean_horizontal(["icoord_2", "icoord_3"]))
        .select("centers").to_series().to_list()
    )
    centers_left = np.array(
        [
            centers[left_matches[i]]
            if left_matches[i] is not None
            else np.nan for i in range(len(left_matches))
        ], dtype=np.float64)
    centers_right = np.array(
        [
            centers[right_matches[i]]
            if right_matches[i] is not None
            else np.nan for i in range(len(right_matches))
            ], dtype=np.float64)

    df = (df.with_columns(
            centers_left=centers_left
        )
        .with_columns(centers_right=centers_right)
        .with_columns(
            pl.when(pl.col("icoord_2").is_nan())
            .then(pl.col("centers_left"))
            .otherwise(pl.col("icoord_2")).alias("icoord_2")
        )
        .with_columns(
            pl.when(pl.col("icoord_3").is_nan())
            .then(pl.col("centers_right"))
            .otherwise(pl.col("icoord_3")).alias("icoord_3")
        )
    )
    return df


def _get_cluster_labels(clusters, labels):

    clusters_df = pl.from_numpy(clusters)

    cluster_labels = (
        clusters_df
        .with_columns(
            label=labels)
        .filter((pl.col("index") == pl.col("cluster_min")) |
                (pl.col("index") == pl.col("cluster_max")))
        .group_by('cluster')
        .agg([
            pl.when(
                pl.col("cluster_min") == pl.col("index")
                ).then(pl.col("label")).alias("start"),
            pl.when(
                pl.col("cluster_max") == pl.col("index")
                ).then(pl.col("label")).alias("end"), pl.first("size")
                ])
        .with_columns(
            pl.when(
                pl.col("end").list.len() < 2
                )
            .then(None).otherwise(pl.col("end")).alias("end")
            )
        .with_columns(
            label=pl.concat_str([
                pl.col('start').list.first(), pl.lit('-'),
                pl.col('end').list.last()],
                ignore_nulls=True)
                )
        .with_columns(
            pl.col("label").str.strip_chars("-").alias("label")
            )
        .with_columns(
            pl.concat_str([
                pl.col("label"), pl.lit('\n('),
                pl.col('size').cast(pl.Utf8),
                pl.lit(')')
                ])
                )
        .sort("start"))

    cluster_labels = cluster_labels.sort(
        "cluster"
        ).get_column("label").to_list()

    return cluster_labels


def _get_cluster_summary(clusters, labels):
    p = np.max(clusters['cluster']).item() + 1
    clusters_df = pl.from_numpy(clusters)

    clusters_df = clusters_df.with_columns(label=labels).with_columns(
        cluster_seq=pl.col("cluster").rle_id()
    )
    cluster_summary = {}
    for i in range(p):
        elements = clusters_df.filter(
            pl.col('cluster_seq') == i
            )['label'].to_list()
        cluster_summary[f"Period {i+1}"] = elements

    return cluster_summary


def _reorder_leaves(R):

    # get list of leaves by index
    ivl_list = R["leaves"]
    color_list = R["color_list"]

    # place the dendrogram coordinates in a dataframe
    coord_df = pl.DataFrame(
        {'icoord': R['icoord'], 'dcoord': R['dcoord']}
        ).with_row_index()

    # unnest the coorinates and sort in order of their placement left-to-right
    coord_df = (coord_df
                .with_columns(
                    pl.col('icoord').list.to_struct(fields=[
                        "icoord_1", "icoord_2", "icoord_3", "icoord_4"]))
                .with_columns(
                    pl.col('dcoord').list.to_struct(fields=[
                        "dcoord_1", "dcoord_2", "dcoord_3", "dcoord_4"]))
                .unnest('icoord')
                .unnest('dcoord')
                ).sort(pl.col("icoord_1"))

    # indicate which leaves are singletons and witch are in pairs
    coord_df = (coord_df
                .with_columns(
                    pl.when(
                        dcoord_1=0, dcoord_4=0
                        ).then(2).otherwise(None).alias("n_leaves")
                )
                .with_columns(
                    pl.when(
                        (pl.col("n_leaves").is_null() &
                            pl.col("dcoord_1").eq(0))
                            ).then(1).otherwise(
                                pl.col("n_leaves")).alias("n_leaves"))
                .with_columns(
                    pl.when(
                        (pl.col("n_leaves").is_null() &
                            pl.col("dcoord_4").eq(0))).then(1).otherwise(
                                pl.col("n_leaves")).alias("n_leaves")))

    # create indices for groupings
    coord_df = (coord_df
                .with_columns(
                    pl.col("n_leaves").cum_sum().sub(2).alias("idx_1")
                )
                .with_columns(
                    pl.when(pl.col("n_leaves").eq(2)).then(1).alias("idx_2")
                )
                .with_columns(
                    pl.col("idx_2").add(pl.col("idx_1"))
                )
                .with_columns(
                    pl.when(
                        pl.col("idx_1").is_not_null() &
                        pl.col("idx_2").is_null()
                        ).then(pl.col("idx_1").add(1)
                               ).otherwise(pl.col("idx_1"))
                )
                ).drop("n_leaves")

    coord_df = (coord_df
                .with_columns(ivl_list=ivl_list)
                .with_columns(
                    ivl_1=pl.col("ivl_list").list.slice(
                        pl.col("idx_1"), length=1).list.first().cast(pl.Int32))
                .with_columns(
                    ivl_2=pl.col("ivl_list").list.slice(
                        pl.col("idx_2"), length=1).list.first().cast(pl.Int32))
                )

    # based on the indices determine how far the iccord of leaves that
    # have icoord_1 or icoord_4 of 0 need to moved to follow
    # the sequential order indicated by ivl integers
    coord_df = (coord_df
                .with_columns(leaf_idx=pl.col("ivl_1").forward_fill())
                .with_columns(
                    pl.col("leaf_idx").shift(-1).alias("next_idx")
                    )
                .with_columns(
                    pl.when(
                        pl.col("next_idx").ne(pl.col("leaf_idx"))
                        ).then(pl.col("next_idx")
                               ).otherwise(None).backward_fill()
                    )
                .with_columns(
                    pl.col("leaf_idx").sub(pl.col("idx_1"))
                    .alias("x_adjust").forward_fill()
                    )
                .with_columns(pl.col("x_adjust").mul(10))
                ).drop("ivl_list")

    # sort by the heights the y-axis so links are ordered
    # bottom to top as indicated by dcoord_2
    coord_df = (coord_df.sort("dcoord_2", descending=False)
                .with_columns(
                    pl.when(
                        pl.col("dcoord_1").eq(0))
                .then(pl.col(["icoord_1", "icoord_2"]).add(pl.col("x_adjust")))
                .otherwise(np.nan)
                )
                .with_columns(
                pl.when(
                    pl.col("dcoord_4").eq(0))
                .then(pl.col(["icoord_3", "icoord_4"]).add(pl.col("x_adjust")))
                .otherwise(np.nan)
                )
                )

    # map linkages by finding the nearest linkages left and right
    # down the dendrogam as indicated by their shared coordinates
    dcoord_1 = coord_df.sort(
        "dcoord_2", descending=False)["dcoord_1"].to_numpy()
    dcoord_2 = coord_df.sort(
        "dcoord_2", descending=False)["dcoord_2"].to_numpy()

    dcoord_3 = coord_df.sort(
        "dcoord_2", descending=False)["dcoord_3"].to_numpy()
    dcoord_4 = coord_df.sort(
        "dcoord_2", descending=False)["dcoord_4"].to_numpy()

    left_matches = []
    for i in range(len(dcoord_1)):
        match = _find_first_previous_match(dcoord_2, dcoord_1[i])
        left_matches.append(match)

    right_matches = []
    for i in range(len(dcoord_1)):
        match = _find_first_previous_match(dcoord_3, dcoord_4[i])
        right_matches.append(match)

    # calculate how much spans need to be shifted
    # along the x-axis by iteratvily updating the
    # centers from the bottom to the top
    df_list = [coord_df]

    for i in range(1, len(right_matches)):
        df = df_list[-1]
        df = _update_centers(df, left_matches, right_matches)
        df_list.append(df)

    # take the last iteration
    coord_df = (
        df_list[-1]
        .with_columns(pl.col("icoord_2").alias("icoord_1"))
        .with_columns(pl.col("icoord_3").alias("icoord_4"))
        .sort("icoord_1")
        )

    # adjust centers to their final position
    coord_df = (
        coord_df
        .with_columns(
            pl.when(
                (pl.col("icoord_3") != pl.col("centers_right")) &
                pl.col("centers_right").is_not_nan()
                )
            .then(
                pl.col("centers_right")
                )
            .otherwise(pl.col("icoord_3")).alias("icoord_3")
                )
        .with_columns(
            pl.when(
                (pl.col("icoord_2") != pl.col("centers_left")) &
                pl.col("centers_left").is_not_nan()
                )
            .then(pl.col("centers_left")
                  )
            .otherwise(pl.col("icoord_2"))
            .alias("icoord_2")
                )
        .with_columns(
            pl.when(
                (pl.col("icoord_4") != pl.col("centers_right")) &
                pl.col("centers_right").is_not_nan()
                )
            .then(pl.col("centers_right"))
            .otherwise(pl.col("icoord_4"))
            .alias("icoord_4")
                )
        .with_columns(
            pl.when(
                (pl.col("icoord_1") != pl.col("centers_left")) &
                pl.col("centers_left").is_not_nan()
                )
            .then(pl.col("centers_left"))
            .otherwise(pl.col("icoord_1"))
            .alias("icoord_1")
            )
            )

    # select columns needed for the updated dendrogram dictionary
    X = (
        coord_df
        .with_columns(
                pl.concat_list(pl.selectors.starts_with("icoord_"))
                .alias('icoord')
                )
        .with_columns(
                pl.concat_list(
                    pl.selectors.starts_with("dcoord_")
                    ).alias('dcoord')
                )
        .with_columns(
                pl.concat_list(
                    pl.selectors.starts_with("ivl_")).alias('ivl')
                ).select("icoord", "dcoord", "ivl")
            )

    # put ivl data into list droping null values
    ivl = [
        x for xs in X.get_column("ivl").to_list() for x in xs if x is not None
        ]

    # format the final dictionary
    X = {"icoord": X.get_column("icoord").to_list(),
         "dcoord": X.get_column("dcoord").to_list(),
         "ivl": ivl,
         "color_list": color_list}

    return X


def _vnc_dendrogram(Z,
                    p=30,
                    truncate_mode=None,
                    color_threshold=0,
                    get_leaves=True,
                    orientation='top',
                    labels=None,
                    count_sort=False,
                    distance_sort=False,
                    show_leaf_counts=True,
                    no_plot=True,
                    no_labels=False,
                    leaf_font_size=None,
                    leaf_rotation=None,
                    leaf_label_func=None,
                    show_contracted=False,
                    link_color_func=None,
                    ax=None,
                    above_threshold_color='k'):
    """
    This is a workaround to access the contraction marks
    to allow for customized plotting.
    """
    xp = sch.array_namespace(Z)
    Z = sch._asarray(Z, order='c', xp=xp)

    if orientation not in ["top", "left", "bottom", "right"]:
        raise ValueError("orientation must be one of 'top', 'left', "
                         "'bottom', or 'right'")

    if labels is not None:
        try:
            len_labels = len(labels)
        except (TypeError, AttributeError):
            len_labels = labels.shape[0]
        if Z.shape[0] + 1 != len_labels:
            raise ValueError("Dimensions of Z and labels must be consistent.")

    sch.is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1

    if isinstance(p, (int, float)):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')

    if p > n or p == 0:
        p = n

    dist = [x[2].item() for x in Z]
    if p > 1 and p < n:
        dist_threshold = np.mean(
            [dist[len(dist)-p+1], dist[len(dist)-p]]
        )
    else:
        dist_threshold = None

    if truncate_mode not in ('lastp', 'mtica', 'level', 'none', None):
        # 'mtica' is kept working for backwards compat.
        raise ValueError('Invalid truncation mode.')

    if truncate_mode == 'mtica':
        # 'mtica' is an alias
        truncate_mode = 'level'

    if truncate_mode == 'level':
        if p <= 0:
            p = np.inf

    if get_leaves:
        lvs = []
    else:
        lvs = None

    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []  # list of leaves

    if color_threshold is None or (isinstance(color_threshold, str) and
                                   color_threshold == 'default'):
        color_threshold = xp.max(Z[:, 2]) * 0.7

    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl,
         'leaves': lvs, 'color_list': color_list}

    # Empty list will be filled in _dendrogram_calculate_info
    contraction_marks = [] if show_contracted else None
    clusters = [] if p > 1 else None

    sch._dendrogram_calculate_info(
        Z=Z, p=p,
        truncate_mode=truncate_mode,
        color_threshold=color_threshold,
        get_leaves=get_leaves,
        orientation=orientation,
        labels=labels,
        count_sort=count_sort,
        distance_sort=distance_sort,
        show_leaf_counts=show_leaf_counts,
        i=2*n - 2,
        iv=0.0,
        ivl=ivl,
        n=n,
        icoord_list=icoord_list,
        dcoord_list=dcoord_list,
        lvs=lvs,
        current_color=current_color,
        color_list=color_list,
        currently_below_threshold=currently_below_threshold,
        leaf_label_func=leaf_label_func,
        contraction_marks=contraction_marks,
        link_color_func=link_color_func,
        above_threshold_color=above_threshold_color)

    if not no_plot:
        mh = xp.max(Z[:, 2])
        sch._plot_dendrogram(icoord_list,
                             dcoord_list,
                             ivl,
                             p,
                             n,
                             mh,
                             orientation,
                             no_labels,
                             color_list,
                             leaf_font_size=leaf_font_size,
                             leaf_rotation=leaf_rotation,
                             contraction_marks=contraction_marks,
                             ax=ax,
                             above_threshold_color=above_threshold_color)

    R["leaves_color_list"] = sch._get_leaves_color_list(R)

    if truncate_mode is None:
        R = _reorder_leaves(R)
        if p > 1:
            clusters_assignment = sch.fcluster(Z, p, criterion='maxclust') - 1
            df_clst = pl.DataFrame({'cluster': clusters_assignment}
                                   ).with_row_index().sort("index")
            df_clst = df_clst.with_columns(
                cluster_min=pl.col("index").min().over(pl.col("cluster"))
                ).with_columns(
                size=pl.col("index").count().over(pl.col("cluster"))
                ).with_columns(
                cluster_max=pl.col("index").max().over(pl.col("cluster"))
                )
            clusters = df_clst.to_numpy(structured=True)
            cluster_labels = _get_cluster_labels(clusters, labels)
            cluster_summary = _get_cluster_summary(clusters, labels)

    if truncate_mode == 'lastp':
        R2 = sch.dendrogram(Z, no_plot=True)
        clusters = R['ivl']
        clusters_assignment = []
        for i in range(len(clusters)):
            if isinstance(clusters[i], str) and clusters[i].startswith("("):
                n = clusters[i].removeprefix("(").removesuffix(")")
                n = int(n)
                clusters_assignment.append([i] * n)
            else:
                clusters_assignment.append([i])
        clusters_assignment = [x for xs in clusters_assignment for x in xs]
        df_clst = pl.DataFrame({'index': R2["leaves"],
                                'cluster': clusters_assignment}).sort("index")
        df_clst = df_clst.with_columns(
            cluster_min=pl.col("index").min().over(pl.col("cluster"))
            ).with_columns(
            size=pl.col("index").count().over(pl.col("cluster"))
            ).with_columns(
            cluster_max=pl.col("index").max().over(pl.col("cluster"))
            )
        clusters = df_clst.to_numpy(structured=True)
        cluster_labels = _get_cluster_labels(clusters, labels)
        cluster_summary = _get_cluster_summary(clusters, labels)

    mh = xp.max(Z[:, 2])
    R['n'] = n
    R['mh'] = mh
    R['p'] = p
    R['labels'] = labels
    R['clusters'] = clusters
    R['cluster_labels'] = cluster_labels
    R['cluster_summary'] = cluster_summary
    R['dist_threshold'] = dist_threshold
    R["contraction_marks"] = contraction_marks

    return R


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
                    cut_line=False,
                    periodize=False) -> Figure:

        dist_types = ['sd', 'cv']
        if distance not in dist_types:
            distance = "sd"
        orientation_types = ['horizontal', 'vertical']
        if orientation not in orientation_types:
            orientation = "horizontal"

        if distance == "cv":
            Z = self.Z_cv
        else:
            Z = self.Z_sd

        if n_periods > len(Z):
            n_periods = 1
            periodize = False

        if n_periods > 1 and n_periods <= len(Z) and periodize is not True:
            cut_line = True

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        # Plot the corresponding dendrogram
        if orientation == "horizontal" and periodize is not True:
            X = _vnc_dendrogram(Z,
                                p=n_periods,
                                labels=self.time_intervals)

            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'],
                                 orientation='top',
                                 p=X['p'],
                                 n=X['n'],
                                 no_labels=False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel(f'Distance (in summed {distance})')
            ax.set_xticklabels(X['labels'],
                               fontsize=font_size,
                               rotation=90)
            plt.setp(ax.collections, linewidth=.5)

            if cut_line and X['dist_threshold'] is not None:
                ax.axhline(y=X['dist_threshold'],
                           color='r',
                           alpha=0.7,
                           linestyle='--',
                           linewidth=.5)

        if orientation == "horizontal" and periodize is True:
            X = _vnc_dendrogram(Z,
                                truncate_mode="lastp",
                                p=n_periods,
                                show_contracted=True,
                                labels=self.time_intervals)
            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'], orientation='top',
                                 p=X['p'],
                                 n=X['n'],
                                 no_labels=False,
                                 contraction_marks=X['contraction_marks'])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_ylabel(f'Distance (in summed {distance})')
            ax.set_xticklabels(X['cluster_labels'],
                               fontsize=font_size,
                               rotation=90)
            plt.setp(ax.collections, linewidth=.5)

        if orientation == "vertical" and periodize is not True:
            X = _vnc_dendrogram(Z,
                                p=n_periods,
                                labels=self.time_intervals)

            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'],
                                 orientation='top',
                                 p=X['p'],
                                 n=X['n'],
                                 no_labels=False)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel(f'Distance (in summed {distance})')
            ax.set_yticklabels(X['labels'],
                               fontsize=font_size,
                               rotation=0)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymax, ymin)
            plt.setp(ax.collections, linewidth=.5)

            if cut_line and X['dist_threshold'] is not None:
                ax.axvline(x=X['dist_threshold'],
                           color='r',
                           alpha=0.7,
                           linestyle='--',
                           linewidth=.5)

        if orientation == "vertical" and periodize is True:
            X = _vnc_dendrogram(Z,
                                truncate_mode="lastp",
                                p=n_periods,
                                show_contracted=True,
                                labels=self.time_intervals)
            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'], orientation='top',
                                 p=X['p'],
                                 n=X['n'],
                                 no_labels=False,
                                 contraction_marks=X['contraction_marks'])

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xlabel(f'Distance (in summed {distance})')
            ax.set_yticklabels(X['cluster_labels'],
                               fontsize=font_size,
                               rotation=0)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymax, ymin)
            plt.setp(ax.collections, linewidth=.5)

        return fig
