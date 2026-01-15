import os

# -------------------------------------------
# Registry of analysis folder names
# (edit / add as needed)
# -------------------------------------------
ANALYSIS_FOLDERS = {
    "FreqBand": "01_Spectrum",
    "FreqTopo": "02_BandTopomaps",
    "FreqTopoNorm": "03_NormBandTopomaps",
    "LBTopo": "04_LobeBandHeatmaps",
    "SVD": "05_SVD_analysis",
    "SS": "07_SS_overall_analysis",
    "SS_Map": "08_SS_Maps",
    "LN": "06_LogNormal_analysis",
    "CoVarMat": "11_CovarianceMatrices",
    "SI_HM": "10_SI_Heatmaps",
    "SI_TM": "09_SI_Topomaps"
}


def get_save_path(analysis_key, patient_id, batch_output_path):
    """
    Returns the full path where a figure should be saved.

    analysis_key       -> short code, e.g. 'SS', 'SVD', 'Entropy'
    patient_id         -> e.g. 'P012'
    batch_output_path  -> .../diff_data2/Results
    """

    # -------------------------------
    # 1) Parent folder name before Results
    #    e.g. diff_data2
    # -------------------------------
    parent_folder = os.path.basename(os.path.dirname(batch_output_path))

    # -------------------------------
    # 2) Determine analysis folder name
    # -------------------------------
    if analysis_key not in ANALYSIS_FOLDERS:
        raise ValueError(f"Unknown analysis key: {analysis_key}")

    analysis_folder_name = ANALYSIS_FOLDERS[analysis_key]

    # -------------------------------
    # 3) Create full analysis folder path
    # -------------------------------
    analysis_dir = os.path.join(batch_output_path, analysis_folder_name)
    os.makedirs(analysis_dir, exist_ok=True)

    # -------------------------------
    # 4) Build filename
    # -------------------------------
    filename = f"{parent_folder}_{patient_id}_{analysis_key}.png"

    # -------------------------------
    # 5) Return complete path
    # -------------------------------
    return os.path.join(analysis_dir, filename)
