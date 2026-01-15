import numpy as np

# --------------------------------------------------
# 1. Channel-wise SS map features
# --------------------------------------------------
def ss_map_ch_features(row_rank_mean, col_rank_mean):
    root2 = np.sqrt(2)

   # ssbi = np.abs(row_rank_mean - col_rank_mean) #row_rank − col_rank

    d_sink = np.sqrt((row_rank_mean - root2)**2 + col_rank_mean**2)
    d_source = np.sqrt(row_rank_mean**2 + (col_rank_mean - root2)**2)

    engagement = np.sqrt(row_rank_mean**2 + col_rank_mean**2)
    efficiency = np.abs(row_rank_mean - col_rank_mean)

    return {
        #"SSBI": ssbi, #row_rank − col_rank
        "d_sink": d_sink, #sqrt((row−√2)^2 + col^2)
        "d_source": d_source, #sqrt(row^2 + (col−√2)^2)
        "engagement": engagement, #sqrt(row^2 + col^2)
        "directional_efficiency": efficiency #|row − col|

    }


# --------------------------------------------------
# 2. Lobe-wise SS map features
# --------------------------------------------------
def ss_map_lb_features(row_rank_mean, col_rank_mean, labels, lobe_map):
    features = {}

    for lobe in set(lobe_map.values()):
        idx = [i for i, ch in enumerate(labels) if lobe_map.get(ch) == lobe]
        if len(idx) == 0:
            continue

        rr = row_rank_mean[idx]
        cr = col_rank_mean[idx]

        centroid = np.array([rr.mean(), cr.mean()])
        points = np.column_stack([rr, cr])

        compactness = np.mean(
            np.linalg.norm(points - centroid, axis=1)
        )

        features[lobe] = {
            "sinkness": rr.mean(),
            "sourceness": cr.mean(),
            "asymmetry": np.mean(rr - cr),
            "compactness": compactness
        }

    return features


# --------------------------------------------------
# 3. Network-wise SS map features
# --------------------------------------------------
def ss_map_nw_features(row_rank_mean, col_rank_mean):
    root2 = np.sqrt(2)

    # cx = np.mean(row_rank_mean)
    # cy = np.mean(col_rank_mean)
    #
    # d_sink_centroid = np.sqrt((cx - root2)**2 + cy**2)
    # d_source_centroid = np.sqrt(cx**2 + (cy - root2)**2)

    q1 = np.mean((row_rank_mean > 0.5) & (col_rank_mean <= 0.5))  # sinks
    q2 = np.mean((row_rank_mean <= 0.5) & (col_rank_mean > 0.5))  # sources
    q3 = np.mean((row_rank_mean > 0.5) & (col_rank_mean > 0.5))   # hubs
    q4 = np.mean((row_rank_mean <= 0.5) & (col_rank_mean <= 0.5))# peripheral

    return {
        # "centroid_x": cx,
        # "centroid_y": cy,
        # "d_sink_centroid": d_sink_centroid,
        # "d_source_centroid": d_source_centroid,
        "quadrant_frac": {
            "sink": q1,
            "source": q2,
            "hub": q3,
            "peripheral": q4
        }
    }
