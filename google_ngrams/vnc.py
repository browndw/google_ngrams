import numpy as np
import pandas as pd


def is_sequence(x, tol=np.sqrt(np.finfo(float).eps), *args):
    if (np.any(np.isnan(x)) or
        np.any(np.isinf(x)) or len(x) <= 1 or
        np.diff(x[:2]) == 0):
        return False
    return np.diff(np.range(np.diff(x))).max() <= tol


def vnc_clust(time,
              values,
              distance_measure="sd"):

    if not isinstance(time, (list, np.ndarray)) or len(time) != len(set(time)):
        raise ValueError("It appears that your time series contains gaps or is not evenly spaced.")
    if len(time) != len(values):
        raise ValueError("Your time and values vectors must be the same length.")
    
    input_values = np.array(values)
    years = np.array(time)
    input_dict = dict(zip(years, input_values))
    
    data_collector = {}
    data_collector["0"] = input_dict
    position_collector = {}
    position_collector[1] = 0
    overall_distance = 0
    number_of_steps = len(input_dict) - 1
    
    for i in range(1, number_of_steps + 1):
        difference_checker = []
        unique_names = list(set(input_dict.keys()))
        
        for j in range(len(unique_names) - 1):
            first_name = unique_names[j]
            second_name = unique_names[j + 1]
            pooled_sample = [input_dict[name] for name in [first_name, second_name] if name in input_dict]
            
            if distance_measure == "sd":
                difference_checker.append(0 if sum(pooled_sample) == 0 else np.std(pooled_sample))
            if distance_measure == "cv":
                difference_checker.append(0 if sum(pooled_sample) == 0 else np.std(pooled_sample) / np.mean(pooled_sample))
        
        pos_to_be_merged = np.argmin(difference_checker)
        distance = min(difference_checker)
        overall_distance += distance
        lower_name = unique_names[pos_to_be_merged]
        higher_name = unique_names[pos_to_be_merged + 1]
        matches = [name for name in input_dict.keys() if name in [lower_name, higher_name]]
        new_mean_age = round(np.mean([float(name) for name in matches]), 4)
        position_collector[i + 1] = [name for name in input_dict.keys() if name in [lower_name, higher_name]]
        input_dict[new_mean_age] = input_dict.pop(lower_name) + input_dict.pop(higher_name)
        data_collector[i + 1] = input_dict.copy()
        data_collector[i + 1]['distance'] = distance
    
    hc_build = pd.DataFrame({
        'start': [min(pos) for pos in position_collector.values()],
        'end': [max(pos) for pos in position_collector.values()]
    })
    
    idx = range(1, len(hc_build) + 1)
    
    y = [np.where(hc_build['start'].iloc[:i-1].values == hc_build['start'].iloc[i-1])[0] for i in idx]
    z = [np.where(hc_build['end'].iloc[:i-1].values == hc_build['end'].iloc[i-1])[0] for i in idx]
    
    merge1 = [np.nanmax(np.where(x == 1)[0]) - 1 if not np.all(np.isnan(x)) else np.nan for x in y]
    merge2 = [np.nanmax(np.where(x == 1)[0]) - 1 if not np.all(np.isnan(x)) else np.nan for x in z]
    
    hc_build['merge1'] = [min(m1, m2) for m1, m2 in zip(merge1, merge2)]
    hc_build['merge2'] = [max(m1, m2) for m1, m2 in zip(merge1, merge2)]
    hc_build['merge2'] = hc_build['merge2'].replace(-np.inf, np.nan)
    
    hc_build['merge1'] = np.where(hc_build['merge1'].isna() & hc_build['merge2'].isna(), -hc_build['start'], hc_build['merge1'])
    hc_build['merge2'] = np.where(hc_build['merge2'].isna(), -hc_build['end'], hc_build['merge2'])
    
    to_merge = [-set(hc_build.iloc[i, 0:2]) - set(hc_build.iloc[1:i-1, 0:2].values.flatten()) for i in idx]
    
    hc_build['merge1'] = np.where(hc_build['merge1'].isna(), to_merge, hc_build['merge1'])
    
    hc_build = hc_build.iloc[1:]
    
    height = np.cumsum([float(name) for name in data_collector.keys() if name != "0"])
    order = list(range(1, len(data_collector) + 1))
    
    m = np.array([hc_build['merge1'].values, hc_build['merge2'].values]).T
    hc = {
        'merge': m,
        'height': height,
        'order': order,
        'labels': years
    }
    
    return hc