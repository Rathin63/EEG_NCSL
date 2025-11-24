import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------
# User settings
# ---------------------------------------------------------

file_A = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\BatchOps\BatchSummary_HC.xlsx"
file_B = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\BatchOps\BatchSummary_ADHD.xlsx"

# Column numbers to compare (1-based Excel index, excluding ID column)
#columns_to_compare = [5, 7, 24] # <<< EDIT HERE

columns_to_compare = [5,10,11,22,23]# <<< EDIT HERE

output_file = r"E:\JHU_Postdoc\Research\EEG\ColumnWiseStats_20Nov.xlsx"



# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
dfA = pd.read_excel(file_A)
dfB = pd.read_excel(file_B)

print(f"Loaded: {file_A} with {len(dfA)} subjects")
print(f"Loaded: {file_B} with {len(dfB)} subjects")

# Convert user input to zero-based index
columns_to_compare = [c - 1 for c in columns_to_compare]

results = []


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def safe_auc(values_A, values_B):
    """Compute AUC using labels; handle constant or NaN cases safely."""
    X = np.concatenate([values_A, values_B])
    y = np.concatenate([np.zeros(len(values_A)), np.ones(len(values_B))])

    # If all values are identical → AUC is meaningless
    if np.nanstd(X) == 0:
        return np.nan

    try:
        auc = roc_auc_score(y, X)
        # Make sure AUC ≥ 0.5 for interpretability
        if auc < 0.5:
            auc = 1 - auc
        return auc
    except:
        return np.nan


def cohen_d(a, b):
    """Classic effect size."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    numerator = np.nanmean(a) - np.nanmean(b)
    denominator = np.sqrt(((np.nanstd(a, ddof=1) ** 2) +
                           (np.nanstd(b, ddof=1) ** 2)) / 2)

    if denominator == 0:
        return np.nan
    return numerator / denominator


# ---------------------------------------------------------
# Main loop: compute stats for each selected feature
# ---------------------------------------------------------
for col in columns_to_compare:

    feature_name = dfA.columns[col]
    A = dfA.iloc[:, col].astype(float)
    B = dfB.iloc[:, col].astype(float)

    # Remove NaNs
    A_clean = A.dropna().values
    B_clean = B.dropna().values

    # Skip empty columns
    if len(A_clean) == 0 or len(B_clean) == 0:
        continue

    # t-test
    t_stat, p_t = ttest_ind(A_clean, B_clean, equal_var=False)

    # Mann-Whitney U
    try:
        _, p_u = mannwhitneyu(A_clean, B_clean, alternative='two-sided')
    except:
        p_u = np.nan

    # Cohen's d
    d = cohen_d(A_clean, B_clean)

    # AUC
    auc = safe_auc(A_clean, B_clean)

    results.append([
        feature_name,
        col + 1,                # Excel-style numbering
        np.nanmean(A_clean),
        np.nanmean(B_clean),
        t_stat,
        p_t,
        p_u,
        d,
        auc
    ])


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------
res_df = pd.DataFrame(results, columns=[
    "Feature",
    "Column_Number",
    "Mean_A",
    "Mean_B",
    "t_stat",
    "p_ttest",
    "p_mannwhitney",
    "Cohens_d",
    "AUC"
])

# FDR correction on t-test p-values
_, p_fdr, _, _ = multipletests(res_df["p_ttest"], method='fdr_bh')
res_df["p_fdr"] = p_fdr

# Ranking
res_df["Rank_AUC"] = res_df["AUC"].rank(ascending=False)
res_df["Rank_d"] = res_df["Cohens_d"].abs().rank(ascending=False)

# Combined ranking (optional)
res_df["Combined_Rank"] = (res_df["Rank_AUC"] + res_df["Rank_d"]) / 2


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
output_file = r"E:\JHU_Postdoc\Research\EEG\Stats_Compare_With_AUC.xlsx"
res_df.to_excel(output_file, index=False)

print("\n====================================================")
print(" Statistical Comparison Completed")
print("----------------------------------------------------")
print(f" Output saved to: {output_file}")
print("====================================================")
print("\nTop features by AUC:")
print(res_df.sort_values("AUC", ascending=False).head(10))
