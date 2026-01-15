def compute_lobe_band_matrices(
    band_power,
    band_power_percent,
    eeg_labels
):
    """
    ----------------------------------------------------------------------
    Compute lobe-wise EEG spectral power (absolute + percentage)

    Rows  = Lobes:
        1) Frontal (Fp, F, FC)
        2) Central–Temporal (T, C, CP)
        3) Parieto–Occipital (P, O)

    Columns = Frequency Bands:
        delta, theta, alpha, beta, gamma

    Inputs
    ------
    band_power : dict
        band_name -> vector length = n_channels
    band_power_percent : dict
        band_name -> vector length = n_channels
    eeg_labels : list
        list of channel labels in same order as data_clean

    Outputs
    -------
    abs_matrix   : (3 × 5) numpy array
    pct_matrix   : (3 × 5) numpy array

    Also prints:
        - channel mapping
        - summary tables
    ----------------------------------------------------------------------
    """

    import numpy as np

    print("\n" + "=" * 70)
    print("COMPUTING LOBE × BAND POWER MATRICES")
    print("=" * 70)

    # -----------------------------
    # 1. Define Lobe Membership
    # -----------------------------
    frontal_prefix  = ("Fp", "F", "FC")
    central_prefix  = ("T", "C", "CP")
    parocc_prefix   = ("P", "O")

    bands_order = ["delta", "theta", "alpha", "beta", "gamma"]
    lobes = ["Frontal", "CentralTemporal", "ParietoOccipital"]

    n_bands = len(bands_order)
    n_lobes = len(lobes)

    abs_matrix = np.zeros((n_lobes, n_bands))
    pct_matrix = np.zeros((n_lobes, n_bands))

    # Track which channels get grouped
    lobe_channels = {l: [] for l in lobes}

    print("\nMapping EEG channels to lobes...\n")

    for ch_idx, ch in enumerate(eeg_labels):

        if ch.startswith(frontal_prefix):
            lobe_channels["Frontal"].append(ch_idx)

        elif ch.startswith(central_prefix):
            lobe_channels["CentralTemporal"].append(ch_idx)

        elif ch.startswith(parocc_prefix):
            lobe_channels["ParietoOccipital"].append(ch_idx)

        else:
            print(f"  [WARNING] Channel {ch} NOT assigned to any lobe")

    # Show mapping
    for l in lobes:
        print(f"{l:20s}: {[eeg_labels[i] for i in lobe_channels[l]]}")

    print("\nComputing lobe-wise sums...\n")

    # ------------------------------------------------
    # 2. Sum Absolute Power Per Lobe/Band
    #   (NO percentage yet)
    # ------------------------------------------------
    for bi, band in enumerate(bands_order):
        bp = band_power[band]

        for li, lobe in enumerate(lobes):
            idxs = lobe_channels[lobe]

            if len(idxs) == 0:
                abs_matrix[li, bi] = np.nan
            else:
                abs_matrix[li, bi] = np.sum(bp[idxs])

    # ------------------------------------------------
    # 3. Convert Absolute Matrix → Global % Matrix
    # ------------------------------------------------
    valid_vals = abs_matrix[~np.isnan(abs_matrix)]
    total_head_power = np.sum(valid_vals)

    pct_matrix = (abs_matrix / total_head_power) * 100.0
    pct_matrix = np.nan_to_num(pct_matrix)

    print(f"\nTOTAL HEAD POWER = {total_head_power:.3f}")
    print(f"CHECK % MATRIX TOTAL = {np.sum(pct_matrix):.2f}%")

    # ------------------------------------------------
    # 4. Print Summary Tables
    # ------------------------------------------------
    print("\nABSOLUTE POWER MATRIX (Rows = Lobes, Columns = Bands)")
    print("Bands:", bands_order)
    print(abs_matrix)

    print("\nPERCENTAGE POWER MATRIX (Rows = Lobes, Columns = Bands)")
    print("Bands:", bands_order)
    print(pct_matrix)

    print("\nCompleted lobe-wise band analysis!\n")

    return abs_matrix, pct_matrix, lobes, bands_order
