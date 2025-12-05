import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# ============================================================
# USER SETTINGS
# ============================================================

file_HC        = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\HC\Results\BatchSummary_HC.xlsx"
file_ADHD_Low  = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Low\Results\BatchSummary_ADHD_Low.xlsx"
file_ADHD_Med  = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Med\Results\BatchSummary_ADHD_Med.xlsx"
file_ADHD_High = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_High\Results\BatchSummary_ADHD_High.xlsx"

# Columns to compare (Excel-style numbering, excluding ID)
columns_to_compare = [
    5, 7, 10, 11, 16, 24, 25,
    *range(27, 40)
]
# columns_to_compare = list(range(5, 27))  # alternative

# Output file (4-group stats)
output_file = r"E:\JHU_Postdoc\Research\EEG\Stats_Compare_4Groups_AUC.xlsx"

# Convert to zero-based indices
columns_to_compare = [c - 1 for c in columns_to_compare]


# ============================================================
# LOAD FILES
# ============================================================

df_dict = {
    "HC":         pd.read_excel(file_HC),
    "ADHD_Low":   pd.read_excel(file_ADHD_Low),
    "ADHD_Med":   pd.read_excel(file_ADHD_Med),
    "ADHD_High":  pd.read_excel(file_ADHD_High),
}

# Print loaded file summary
for name, df in df_dict.items():
    print(f"Loaded {name}: {len(df)} subjects")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_auc(values_A, values_B):
    """
    Safely calculate AUC; avoid errors if constant values or NaNs.
    Ensures AUC >= 0.5 by flipping if needed.
    """
    values_A = np.asarray(values_A, float)
    values_B = np.asarray(values_B, float)

    X = np.concatenate([values_A, values_B])
    y = np.concatenate([np.zeros(len(values_A)), np.ones(len(values_B))])

    if np.nanstd(X) == 0:  # No variation → AUC meaningless
        return np.nan

    try:
        auc = roc_auc_score(y, X)
        if auc < 0.5:
            auc = 1 - auc  # Flip for interpretability
        return auc
    except Exception:
        return np.nan


def cohen_d(a, b):
    """
    Effect size: Cohen's d (Welch-style pooled SD).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    num = np.nanmean(a) - np.nanmean(b)
    pooled = np.sqrt(((np.nanstd(a, ddof=1) ** 2) +
                      (np.nanstd(b, ddof=1) ** 2)) / 2)

    return np.nan if pooled == 0 else num / pooled


# ============================================================
# MAIN LOOP — COMPARE HC VS EACH ADHD SUBGROUP
# ============================================================

results = []

for col in columns_to_compare:

    # Feature name based on first dataframe
    feature_name = list(df_dict.values())[0].columns[col]

    # Extract clean numeric values from all groups
    groups = {
        gname: df.iloc[:, col].dropna().astype(float).values
        for gname, df in df_dict.items()
    }

    # If any group is empty for this feature → skip
    if any(len(v) == 0 for v in groups.values()):
        print(f"Skipping {feature_name} (column {col+1}): at least one group is empty.")
        continue

    # Perform comparisons: HC vs each severity level
    for target_group in ["ADHD_Low", "ADHD_Med", "ADHD_High"]:

        A_clean = groups["HC"]
        B_clean = groups[target_group]

        # Welch's t-test
        t_stat, p_t = ttest_ind(A_clean, B_clean, equal_var=False)

        # Mann-Whitney U test
        try:
            _, p_u = mannwhitneyu(A_clean, B_clean, alternative='two-sided')
        except Exception:
            p_u = np.nan

        # Effect size and AUC
        d_val = cohen_d(A_clean, B_clean)
        auc_val = safe_auc(A_clean, B_clean)

        # Append row
        results.append([
            feature_name,
            col + 1,           # Excel column number
            "HC",
            target_group,
            np.nanmean(A_clean),
            np.nanmean(B_clean),
            t_stat,
            p_t,
            p_u,
            d_val,
            auc_val
        ])

# ============================================================
# CREATE FINAL DATAFRAME (4-GROUP COMPARISONS)
# ============================================================

res_df = pd.DataFrame(results, columns=[
    "Feature",
    "Column_Number",
    "Group_A",
    "Group_B",
    "Mean_A (HC)",
    "Mean_B (Target)",
    "t_stat",
    "p_ttest",
    "p_mannwhitney",
    "Cohens_d",
    "AUC"
])

# FDR correction for t-tests
if len(res_df) > 0:
    _, p_fdr, _, _ = multipletests(res_df["p_ttest"], method='fdr_bh')
    res_df["p_fdr"] = p_fdr
else:
    res_df["p_fdr"] = np.nan

# Rankings
res_df["Rank_AUC"] = res_df["AUC"].rank(ascending=False)
res_df["Rank_d"] = res_df["Cohens_d"].abs().rank(ascending=False)
res_df["Combined_Rank"] = (res_df["Rank_AUC"] + res_df["Rank_d"]) / 2

# Save 4-group comparison
res_df.to_excel(output_file, index=False)

print("\nSaved 4-group stats to:", output_file)


# ============================================================
# SECTION 1: FEATURE-LEVEL SUMMARY TABLE (ACROSS ADHD SUBTYPES)
# ============================================================

feature_summary = (
    res_df
    .groupby("Feature")
    .agg(
        mean_AUC=("AUC", "mean"),
        max_AUC=("AUC", "max"),
        mean_abs_d=("Cohens_d", lambda x: x.abs().mean()),
        best_Combined_Rank=("Combined_Rank", "min")
    )
    .sort_values("best_Combined_Rank")
    .reset_index()
)

summary_file = output_file.replace(".xlsx", "_FeatureSummary.xlsx")
feature_summary.to_excel(summary_file, index=False)
print(f"\nFeature-level summary saved to: {summary_file}")

print("\nTop features (by best Combined_Rank):")
print(feature_summary.head(10))


# ============================================================
# SECTION 1B: 2-CLASS FEATURE SUMMARY (HC vs ADHD_All)
# ============================================================

# Combine all ADHD subtypes into a single group
df_ADHD_all = pd.concat(
    [df_dict["ADHD_Low"], df_dict["ADHD_Med"], df_dict["ADHD_High"]],
    ignore_index=True
)

two_class_rows = []

for col in columns_to_compare:
    feature_name = list(df_dict.values())[0].columns[col]

    HC_vals = df_dict["HC"].iloc[:, col].dropna().astype(float).values
    ADHD_vals = df_ADHD_all.iloc[:, col].dropna().astype(float).values

    if len(HC_vals) == 0 or len(ADHD_vals) == 0:
        continue

    t_stat, p_t = ttest_ind(HC_vals, ADHD_vals, equal_var=False)
    try:
        _, p_u = mannwhitneyu(HC_vals, ADHD_vals, alternative="two-sided")
    except Exception:
        p_u = np.nan

    d_val = cohen_d(HC_vals, ADHD_vals)
    auc_val = safe_auc(HC_vals, ADHD_vals)

    two_class_rows.append([
        feature_name,
        col + 1,
        np.nanmean(HC_vals),
        np.nanmean(ADHD_vals),
        t_stat,
        p_t,
        p_u,
        d_val,
        auc_val
    ])

df_two_class = pd.DataFrame(two_class_rows, columns=[
    "Feature",
    "Column_Number",
    "Mean_HC",
    "Mean_ADHD_All",
    "t_stat",
    "p_ttest",
    "p_mannwhitney",
    "Cohen_d",
    "AUC"
])

if len(df_two_class) > 0:
    _, p_fdr2, _, _ = multipletests(df_two_class["p_ttest"], method="fdr_bh")
    df_two_class["p_fdr"] = p_fdr2
else:
    df_two_class["p_fdr"] = np.nan

df_two_class["Rank_AUC"] = df_two_class["AUC"].rank(ascending=False)
df_two_class["Rank_d"] = df_two_class["Cohen_d"].abs().rank(ascending=False)
df_two_class["Combined_Rank"] = (df_two_class["Rank_AUC"] + df_two_class["Rank_d"]) / 2

two_class_file = output_file.replace(".xlsx", "_TwoClass_HC_vs_ADHDAll.xlsx")
df_two_class.to_excel(two_class_file, index=False)
print(f"\n2-class (HC vs ADHD_All) feature summary saved to: {two_class_file}")

print("\nTop 10 features (HC vs ADHD_All by Combined_Rank):")
print(df_two_class.sort_values("Combined_Rank").head(10))


# ============================================================
# SECTION 2: HEATMAPS (AUC + |Cohen's d|)
# ============================================================

auc_pivot = res_df.pivot_table(
    index="Feature", columns="Group_B", values="AUC", aggfunc="mean"
)

d_pivot = res_df.pivot_table(
    index="Feature", columns="Group_B", values="Cohens_d",
    aggfunc=lambda x: x.abs().mean()
)

# --- AUC heatmap ---
plt.figure(figsize=(8, max(4, 0.4 * len(auc_pivot))))
plt.imshow(auc_pivot.values, aspect="auto")
plt.colorbar(label="AUC")
plt.xticks(range(auc_pivot.shape[1]), auc_pivot.columns, rotation=45, ha="right")
plt.yticks(range(auc_pivot.shape[0]), auc_pivot.index)
plt.title("AUC Heatmap: HC vs ADHD subgroups")
plt.tight_layout()
plt.show()

# --- |Cohen's d| heatmap ---
plt.figure(figsize=(8, max(4, 0.4 * len(d_pivot))))
plt.imshow(d_pivot.values, aspect="auto")
plt.colorbar(label="|Cohen's d|")
plt.xticks(range(d_pivot.shape[1]), d_pivot.columns, rotation=45, ha="right")
plt.yticks(range(d_pivot.shape[0]), d_pivot.index)
plt.title("|Cohen's d| Heatmap: HC vs ADHD subgroups")
plt.tight_layout()
plt.show()

print("\n=== Mean AUC matrix (Feature × ADHD subgroup) ===")
print(auc_pivot)

print("\n=== Mean |Cohen's d| matrix (Feature × ADHD subgroup) ===")
print(d_pivot)


# ============================================================
# SECTION 4: ADVANCED BOXPLOTS (4-box + 2-box)
# ============================================================

# Colors
four_colors = {
    "HC":        "#2ECC71",   # Green
    "ADHD_Low":  "#FF69B4",   # Pink
    "ADHD_Med":  "#FF9999",   # Light Red
    "ADHD_High": "#800000",   # Maroon
}

two_colors = {
    "HC":        "#2ECC71",   # Green
    "ADHD_All":  "#E74C3C",   # Red
}

# Helper: get column index by feature name
base_df = list(df_dict.values())[0]
def get_col_index(feature_name: str) -> int:
    return base_df.columns.get_loc(feature_name)

# List of features to plot (from stats table)
features_for_boxplot = res_df["Feature"].unique().tolist()

for feat in features_for_boxplot:

    col_idx = get_col_index(feat)

    # --------------------------
    # Collect values for 4 groups
    # --------------------------
    group_order = ["HC", "ADHD_Low", "ADHD_Med", "ADHD_High"]

    data_to_plot = []
    for g in group_order:
        vals = df_dict[g].iloc[:, col_idx].dropna().astype(float).values
        data_to_plot.append(vals)

    # Skip if any group empty
    if any(len(v) == 0 for v in data_to_plot):
        print(f"Skipping boxplots for {feat}: at least one group empty.")
        continue

    # ============================================================
    # 4-BOX PLOT: HC vs Low vs Med vs High
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data_to_plot, labels=group_order, showfliers=False)

    # Plot datapoints + means
    for i, gname in enumerate(group_order, start=1):
        vals = data_to_plot[i-1]
        color = four_colors[gname]

        jitter = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(jitter, vals, facecolors='none', edgecolors=color, alpha=0.7)
        ax.scatter(i, np.mean(vals), facecolors=color, edgecolors='k', s=90, zorder=3)

    # Dynamic Y-limits
    all_vals = np.concatenate(data_to_plot)
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    span = y_max - y_min
    pad = 0.15 * span if span != 0 else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)

    # Overall mean
    overall_mean = np.mean(all_vals)
    ax.axhline(overall_mean, linestyle='--', color='k', alpha=0.3)
    ax.scatter(len(group_order) + 0.2, overall_mean, color='k', marker='D', s=90)

    ax.set_title(f"{feat}: HC vs ADHD (4-group)")
    ax.set_ylabel(feat)

    # p-values
    p_lines = []
    for (idx1, g1), (idx2, g2) in combinations(list(enumerate(group_order)), 2):
        v1, v2 = data_to_plot[idx1], data_to_plot[idx2]
        _, p_val = ttest_ind(v1, v2, equal_var=False)
        p_lines.append(f"{g1} vs {g2}: p={p_val:.3e}")

    ax.text(
        0.02, 0.98, "\n".join(p_lines),
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=7,
        bbox=dict(boxstyle="round", alpha=0.25),
    )

    plt.tight_layout()
    plt.show()

    # ============================================================
    # 2-BOX PLOT: HC vs ADHD Combined
    # ============================================================

    fig, ax = plt.subplots(figsize=(6, 4))

    ADHD_all_vals = np.concatenate(data_to_plot[1:])  # combine Low+Med+High
    two_data = [data_to_plot[0], ADHD_all_vals]
    two_labels = ["HC", "ADHD_All"]
    two_colors_list = [two_colors["HC"], two_colors["ADHD_All"]]

    ax.boxplot(two_data, labels=two_labels, showfliers=False)

    # datapoints + means
    for i, (vals, color) in enumerate(zip(two_data, two_colors_list), start=1):
        jitter = np.random.normal(i, 0.04, size=len(vals))
        ax.scatter(jitter, vals, facecolors='none', edgecolors=color, alpha=0.7)
        ax.scatter(i, np.mean(vals), facecolors=color, edgecolors='k', s=90, zorder=3)

    # Dynamic Y-limits
    vals_all = np.concatenate(two_data)
    y_min2, y_max2 = np.min(vals_all), np.max(vals_all)
    span2 = y_max2 - y_min2
    pad2 = 0.15 * span2 if span2 != 0 else 1.0
    ax.set_ylim(y_min2 - pad2, y_max2 + pad2)

    # Overall mean across both
    overall2 = np.mean(vals_all)
    ax.axhline(overall2, linestyle='--', color='k', alpha=0.3)
    ax.scatter(2.25, overall2, color='k', marker='D', s=90)

    ax.set_title(f"{feat}: HC vs ADHD Combined")
    ax.set_ylabel(feat)

    # p-value HC vs ADHD_All
    _, p_main = ttest_ind(two_data[0], two_data[1], equal_var=False)
    ax.text(
        0.02, 0.98, f"HC vs ADHD_All: p={p_main:.3e}",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", alpha=0.25),
    )

    plt.tight_layout()
    plt.show()


print("\n=================================================")
print(" 4-GROUP + 2-CLASS Statistical Comparison Completed")
print("-------------------------------------------------")
print(f" 4-group stats file      : {output_file}")
print(f" Feature summary file    : {summary_file}")
print(f" 2-class summary file    : {two_class_file}")
print("=================================================\n")

print("Top features by AUC (all rows):")
print(res_df.sort_values("AUC", ascending=False).head(10))
