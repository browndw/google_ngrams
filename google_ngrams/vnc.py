import copy
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from textwrap import dedent
from matplotlib.figure import Figure
from scipy.cluster import hierarchy as sch


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
        data_collector[f"{i}: {distance}"] = input_values

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
        .with_columns(pl.col("distance").str.replace(r"(\d+: )", ""))
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


def _contract_linkage_matrix(Z: np.ndarray,
                             p=4):
    """
    Contracts the linkage matrix by reducing the number of clusters
    to a specified number.

    Parameters
    ----------
    Z : np.ndarray
        The linkage matrix.
    p : int
        The number of clusters to retain.

    Returns
    -------
    np.ndarray
        The contracted linkage matrix with updated cluster IDs
        and member counts.
    """
    Z = Z.copy()
    truncated_Z = Z[-(p - 1):]

    n_points = Z.shape[0] + 1
    clusters = [
        dict(node_id=i, left=i, right=i, members=[i], distance=0, n_members=1)
        for i in range(n_points)
    ]
    for z_i in range(Z.shape[0]):
        row = Z[z_i]
        left = int(row[0])
        right = int(row[1])
        cluster = dict(
            node_id=z_i + n_points,
            left=left,
            right=right,
            members=[],
            distance=row[2],
            n_members=int(row[3])
        )
        cluster["members"].extend(copy.deepcopy(clusters[left]["members"]))
        cluster["members"].extend(copy.deepcopy(clusters[right]["members"]))
        cluster["members"].sort()
        clusters.append(cluster)

    node_map = []
    for i in range(truncated_Z.shape[0]):
        node_ids = [int(truncated_Z[i, 0]), int(truncated_Z[i, 1])]
        for cluster in clusters:
            if cluster['node_id'] in node_ids:
                node_map.append(cluster)

    filtered_node_map = []
    superset_node_map = []

    for node in node_map:
        is_superset = False
        for other_node in node_map:
            if (
                node != other_node
                    and set(
                        node['members']
                        ).issuperset(set(other_node['members']))
                    ):
                is_superset = True
                break
        if is_superset:
            superset_node_map.append(node)
        else:
            filtered_node_map.append(node)

    # Add 'truncated_id' to each dictionary in filtered_node_map
    for idx, node in enumerate(
        sorted(filtered_node_map, key=lambda x: x['members'][0])
            ):
        node['truncated_id'] = idx
        node['n_members'] = 1

    for idx, node in enumerate(
        sorted(superset_node_map, key=lambda x: x['node_id'])
            ):
        node['truncated_id'] = idx + len(filtered_node_map)

    # Adjust 'n_members' in superset_node_map to reflect
    # the number of filtered_node_map['members'] sets they contain
    for superset_node in superset_node_map:
        count = 0
        for filtered_node in filtered_node_map:
            if set(
                filtered_node['members']
                    ).issubset(set(superset_node['members'])):
                count += 1
        superset_node['n_members'] = count

    # Create a mapping from node_id to truncated_id and n_members
    node_id_to_truncated_id = {
        node['node_id']: node['truncated_id']
        for node in filtered_node_map + superset_node_map
    }
    node_id_to_n_members = {
        node['node_id']: node['n_members']
        for node in filtered_node_map + superset_node_map
    }

    # Replace values in truncated_Z
    for i in range(truncated_Z.shape[0]):
        truncated_Z[i, 3] = (
            node_id_to_n_members[int(truncated_Z[i, 0])] +
            node_id_to_n_members[int(truncated_Z[i, 1])]
        )
        truncated_Z[i, 0] = node_id_to_truncated_id[int(truncated_Z[i, 0])]
        truncated_Z[i, 1] = node_id_to_truncated_id[int(truncated_Z[i, 1])]

    return truncated_Z


def _contraction_mark_coordinates(Z: np.ndarray,
                                  p=4):
    """
    Generates contraction marks for a given linkage matrix.

    Parameters
    ----------
    Z : np.ndarray
        The linkage matrix.
    p : int
        The number of clusters to retain.

    Returns
    -------
    list
        A sorted list of tuples where each tuple contains
        a calculated value based on truncated_id and a distance value.
    """
    Z = Z.copy()
    truncated_Z = Z[-(p-1):]

    n_points = Z.shape[0] + 1
    clusters = [dict(node_id=i,
                     left=i,
                     right=i,
                     members=[i],
                     distance=0,
                     n_members=1) for i in range(n_points)]
    for z_i in range(Z.shape[0]):
        row = Z[z_i]
        left = int(row[0])
        right = int(row[1])
        cluster = dict(
            node_id=z_i + n_points,
            left=left, right=right,
            members=[],
            distance=row[2],
            n_members=int(row[3])
            )
        cluster["members"].extend(copy.deepcopy(clusters[left]["members"]))
        cluster["members"].extend(copy.deepcopy(clusters[right]["members"]))
        cluster["members"].sort()
        clusters.append(cluster)

    node_map = []
    for i in range(truncated_Z.shape[0]):
        node_ids = [int(truncated_Z[i, 0]), int(truncated_Z[i, 1])]
        for cluster in clusters:
            if cluster['node_id'] in node_ids:
                node_map.append(cluster)

    filtered_node_map = []
    superset_node_map = []

    for node in node_map:
        is_superset = False
        for other_node in node_map:
            if (node != other_node
                    and set(node['members']
                            ).issuperset(set(other_node['members']))):
                is_superset = True
                break
        if is_superset:
            superset_node_map.append(node)
        else:
            filtered_node_map.append(node)

    # Create a set of node_ids from filtered_node_map and superset_node_map
    excluded_node_ids = set(
        node['node_id'] for node in filtered_node_map
            ).union(node['node_id'] for node in superset_node_map)

    # Filter clusters that are not in excluded_node_ids
    non_excluded_clusters = [
        cluster for cluster in clusters
        if cluster['node_id'] not in excluded_node_ids
        ]

    # Create a list to store the result
    subset_clusters = []

    # Iterate over filtered_node_map
    for filtered_cluster in filtered_node_map:
        distances = []
        for cluster in non_excluded_clusters:
            if (
                cluster['n_members'] > 1
                    and set(cluster['members']
                            ).issubset(set(filtered_cluster['members']))):
                distances.append(cluster['distance'])
        if distances:
            subset_clusters.append(
                {'node_id': filtered_cluster['node_id'], 'distance': distances}
                )

    # Add 'truncated_id' to each dictionary in filtered_node_map
    for idx, node in enumerate(
        sorted(filtered_node_map, key=lambda x: x['members'][0])
            ):
        node['truncated_id'] = idx

    # Create a mapping from node_id to truncated_id
    node_id_to_truncated_id = {
        node['node_id']: node['truncated_id'] for node in filtered_node_map
        }

    # Add 'truncated_id' to each dictionary in subset_clusters
    for cluster in subset_clusters:
        cluster['truncated_id'] = node_id_to_truncated_id[cluster['node_id']]

    # Create a list of tuples
    contraction_marks = []

    # Iterate over subset_clusters
    for cluster in subset_clusters:
        truncated_id = cluster['truncated_id']
        for distance in cluster['distance']:
            contraction_marks.append((10.0 * truncated_id + 5.0, distance))

    # Sort the list of tuples
    contraction_marks = sorted(contraction_marks, key=lambda x: (x[0], x[1]))

    return contraction_marks


def _convert_linkage_to_coordinates(Z: np.ndarray):
    """
    Converts a linkage matrix to coordinates for plotting a dendrogram.

    Parameters
    ----------
    Z : np.ndarray
        The linkage matrix.

    Returns
    -------
    dict
        A dictionary containing 'icoord', 'dcoord', and 'ivl'
        for plotting the dendrogram.
    """
    ivl = [i for i in range(Z.shape[0] + 1)]
    n = len(ivl)
    icoord = []
    dcoord = []
    clusters = {i: [i] for i in range(n)}
    current_index = n
    positions = {i: (i + 1) * 10 - 5 for i in range(n)}
    heights = {i: 0 for i in range(n)}

    for i in range(len(Z)):
        cluster1 = int(Z[i, 0])
        cluster2 = int(Z[i, 1])
        dist = Z[i, 2].item()
        new_cluster = clusters[cluster1] + clusters[cluster2]
        clusters[current_index] = new_cluster

        x1 = positions[cluster1]
        x2 = positions[cluster2]
        x_new = (x1 + x2) / 2
        positions[current_index] = x_new

        h1 = heights[cluster1]
        h2 = heights[cluster2]
        heights[current_index] = dist

        icoord.append([x1, x1, x2, x2])
        dcoord.append([h1, dist, dist, h2])

        current_index += 1

    # Sort icoord and dcoord by the first element in each icoord list
    sorted_indices = sorted(range(len(icoord)), key=lambda i: icoord[i][0])
    icoord = [icoord[i] for i in sorted_indices]
    dcoord = [dcoord[i] for i in sorted_indices]

    return {"icoord": icoord, "dcoord": dcoord, "ivl": ivl}


def _vnc_calculate_info(Z: np.ndarray,
                        p=None,
                        truncate=False,
                        contraction_marks=False,
                        labels=None):
    Z = Z.copy()
    Zs = Z.shape
    n = Zs[0] + 1

    if labels is not None:
        if Zs[0] + 1 != len(labels):
            labels = None
            print(dedent(
                """
                Dimensions of Z and labels are not consistent.
                Using defalut labels.
                """))
    if labels is None:
        labels = [str(i) for i in range(Zs[0] + 1)]
    else:
        labels = labels

    if p is not None and p > n or p < 2:
        p = None
        truncate = False
        contraction_marks = False

    if p is not None:
        cluster_assignment = [i.item() for i in sch.cut_tree(Z, p)]

        # Create a dictionary to hold the clusters
        cluster_dict = {}

        # Iterate over the labels and clusters to populate the dictionary
        for label, cluster in zip(labels, cluster_assignment):
            cluster_key = f'cluster_{cluster + 1}'
            if cluster_key not in cluster_dict:
                cluster_dict[cluster_key] = []
            cluster_dict[cluster_key].append(label)

        # Convert the dictionary to a list of dictionaries
        cluster_list = [{key: value} for key, value in cluster_dict.items()]

        # Create a new list to hold the cluster labels
        cluster_labels = []

        # Iterate over the cluster_list to create the labels
        for cluster in cluster_list:
            for key, value in cluster.items():
                if len(value) == 1:
                    cluster_labels.append(str(value[0]))
                else:
                    cluster_labels.append(f"{value[0]}-{value[-1]}")

        # get distance for plotting cut line
        dist = [x[2].item() for x in Z]
        dist_threshold = np.mean(
            [dist[len(dist)-p+1], dist[len(dist)-p]]
        )
    else:
        dist_threshold = None
        cluster_list = None
        cluster_labels = None

    if truncate is True:
        truncated_Z = _contract_linkage_matrix(Z, p=p)

        if contraction_marks is True:
            contraction_marks = _contraction_mark_coordinates(Z, p=p)
        else:
            contraction_marks = None

        Z = truncated_Z
    else:
        Z = Z
        contraction_marks = None

    R = _convert_linkage_to_coordinates(Z)

    mh = np.max(Z[:, 2])
    Zn = Z.shape[0] + 1
    color_list = ['k'] * (Zn - 1)
    leaves_color_list = ['k'] * Zn
    R['n'] = Zn
    R['mh'] = mh
    R['p'] = p
    R['labels'] = labels
    R['color_list'] = color_list
    R['leaves_color_list'] = leaves_color_list
    R['clusters'] = cluster_list
    R['cluster_labels'] = cluster_labels
    R['dist_threshold'] = dist_threshold
    R["contraction_marks"] = contraction_marks

    return R


def _lowess(x,
            y,
            f=1./3.):
    """
    Basic LOWESS smoother with uncertainty.
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors.
        - No higher order polynomials d=1
            so linear smoother.
    """
    # get some paras
    # effective width after reduction factor
    xwidth = f*(x.max()-x.min())
    # number of obs
    N = len(x)
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d: np.clip((1 - np.abs(d)**3)**3, 0, 1)  # noqa: E731
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        # equiv of A.dot(yest) just for k
        yest = A[i].dot(sol)
        place = order[i]
        y_sm[place] = yest
        sigma2 = (np.sum((A.dot(sol) - y[order])**2)/N)
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 *
                                  A[i].dot(np.linalg.inv(ATA)
                                           ).dot(A[i]))
    return y_sm, y_stderr


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

    def timeviz_barplot(self,
                        width=8,
                        height=4,
                        dpi=150,
                        barwidth=4,
                        fill_color='#440154',
                        tick_interval=None,
                        label_rotation=None):
        """
        Generate a bar plot of token frequenices over time.

        Parameters
        ----------
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.
        barwidth:
            The width of the bars.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        xx = self.time_intervals
        yy = self.frequencies

        if label_rotation is None:
            rotation = 90
        else:
            rotation = label_rotation

        if tick_interval is None:
            interval = np.diff(xx)[0]
        else:
            interval = tick_interval

        start_value = np.min(xx)

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.bar(xx, yy, color=fill_color, edgecolor='black',
               linewidth=.5, width=barwidth)
        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=.5)

        ax.tick_params(axis="x", which="both", labelrotation=rotation)
        ax.grid(axis='y', color='w', linestyle='--', linewidth=.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(base=interval,
                                                       offset=start_value))

        return fig

    def timeviz_scatterplot(self,
                            width=8,
                            height=4,
                            dpi=150,
                            point_color='black',
                            point_size=0.5,
                            ci='standard'):
        """
        Generate a bar plot of token frequenices over time.

        Parameters
        ----------
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.
        point_size:
            The size of the points.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        ci_types = ['standard', 'strict', 'both']
        if ci not in ci_types:
            ci = "standard"

        xx = self.time_intervals
        yy = self.frequencies

        order = np.argsort(xx)

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        # run it
        y_sm, y_std = _lowess(xx, yy, f=1./5.)
        # plot it
        ax.plot(xx[order], y_sm[order],
                color='tomato', linewidth=.5, label='LOWESS')
        if ci == 'standard':
            ax.fill_between(
                xx[order], y_sm[order] - 1.96*y_std[order],
                y_sm[order] + 1.96*y_std[order], alpha=0.3,
                label='95 uncertainty')
        if ci == 'strict':
            ax.fill_between(
                xx[order], y_sm[order] - y_std[order],
                y_sm[order] + y_std[order], alpha=0.3,
                label='97.5 uncertainty')
        if ci == 'both':
            ax.fill_between(
                xx[order], y_sm[order] - 1.96*y_std[order],
                y_sm[order] + 1.96*y_std[order], alpha=0.3,
                label='95 uncertainty')
            ax.fill_between(
                xx[order], y_sm[order] - y_std[order],
                y_sm[order] + y_std[order], alpha=0.3,
                label='97.5 uncertainty')

        ax.scatter(xx, yy, s=point_size, color=point_color, alpha=0.75)

        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ticks = [tick for tick in plt.gca().get_yticks() if tick >= 0]
        plt.gca().set_yticks(ticks)

        return fig

    def timeviz_screeplot(self,
                          distance="sd",
                          width=6,
                          height=3,
                          dpi=150,
                          point_size=0.75,):
        """Generate a scree plot for determining clusters.

        Parameters
        ----------
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.
        mda:
            Whether or not non-colinear features should be
            filter out per Biber's multi-dimensional analysis procedure.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        dist_types = ['sd', 'cv']
        if distance not in dist_types:
            distance = "sd"

        if distance == "cv":
            dist = self.distances_cv
        else:
            dist = self.distances_sd

        # SCREEPLOT
        yy = dist[::-1]
        xx = np.array([i for i in range(1, len(yy) + 1)])
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.scatter(x=xx,
                   y=yy,
                   marker='o',
                   s=point_size,
                   facecolors='none',
                   edgecolors='black')
        ax.set_xlabel('Clusters')
        ax.set_ylabel(f'Distance (in summed {distance})')

        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return fig

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
            X = _vnc_calculate_info(Z,
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
            X = _vnc_calculate_info(Z,
                                    truncate=True,
                                    p=n_periods,
                                    contraction_marks=True,
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
            X = _vnc_calculate_info(Z,
                                    p=n_periods,
                                    labels=self.time_intervals)

            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'],
                                 orientation='right',
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
            X = _vnc_calculate_info(Z,
                                    truncate=True,
                                    p=n_periods,
                                    contraction_marks=True,
                                    labels=self.time_intervals)
            sch._plot_dendrogram(icoords=X['icoord'],
                                 dcoords=X['dcoord'],
                                 ivl=X['ivl'],
                                 color_list=X['color_list'],
                                 mh=X['mh'], orientation='right',
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
