import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_ind

def compute_vif_subset(Z_sub):
    """
    Compute VIF only on the subset of features being considered.
    VIF must be conditional on the selected set (not global).
    """
    return pd.Series(
        [variance_inflation_factor(Z_sub.values, i)
         for i in range(Z_sub.shape[1])],
        index=Z_sub.columns
    )


# ================================================================
#  FEATURE SELECTION (mRMR + Interaction Gain + VIF + Correlation)
# ================================================================
def select_optimal_features(X_df, y,
                            corr_thresh=0.75,
                            vif_soft=12,
                            vif_hard=6,
                            top_k=6,
                            weights=(0.20, 0.35, 0.30, 0.15),  # (w_MI, w_AUC, w_D, w_IG)
                            return_selected_only=False,
                            verbose=True
                            ):
    """
    Unified feature selection:
        ✓ Mutual Information (relevance)
        ✓ Univariate AUC + Cohen d
        ✓ Interaction Gain (pair synergy)
        ✓ Redundancy control (Pearson corr)
        ✓ Multicollinearity control (VIF relaxed for top features)

    Returns:
        selected : list of chosen features
        report   : detailed scoring table
        auc_plot : bar chart of feature-wise AUC
    """

    print("\n================ FEATURE SELECTION =================")

    # ---------------------------------------------------
    # 1) Standardize
    # ---------------------------------------------------
    Z = StandardScaler().fit_transform(X_df)
    Z = pd.DataFrame(Z, columns=X_df.columns)


    # ---------------------------------------------------
    # 2) Mutual Information (captures nonlinear relevance)
    # ---------------------------------------------------
    print("\nComputing mutual information...")
    mi = mutual_info_classif(Z, y, random_state=42)
    mi = pd.Series(mi, index=Z.columns, name="MI")


    # ---------------------------------------------------
    # 3) Univariate discrimination metrics
    # ---------------------------------------------------
    print("\nComputing effect size & AUC...")
    rows = []

    for f in Z.columns:
        x1 = Z.loc[y==0, f]
        x2 = Z.loc[y==1, f]

        d = (x1.mean()-x2.mean())/np.sqrt((x1.var()+x2.var())/2)
        auc = roc_auc_score(y, Z[f])
        _, p = ttest_ind(x1, x2)

        rows.append((f, abs(d), auc, p))

    stats = pd.DataFrame(rows,
                         columns=["feature","cohen_d","auc","p_value"])


    # ---------------------------------------------------
    # 4) Interaction Gain (pair synergy)
    # ---------------------------------------------------
    print("\nEstimating interaction gain...")
    ig = {}

    for i in range(len(Z.columns)):
        for j in range(i+1, len(Z.columns)):

            f1 = Z.columns[i]
            f2 = Z.columns[j]

            # best single AUC
            best_single = max(roc_auc_score(y, Z[f1]),
                              roc_auc_score(y, Z[f2]))

            # logistic 2-feature model
            model = LogisticRegression(max_iter=500)
            model.fit(Z[[f1,f2]], y)

            auc_pair = roc_auc_score(
                y,
                model.predict_proba(Z[[f1,f2]])[:,1]
            )

            ig[(f1,f2)] = auc_pair - best_single

    ig = pd.Series(ig, name="IG").sort_values(ascending=False)

    # aggregate IG per-feature
    ig_feat = pd.Series(0.0, index=Z.columns)
    for (f1,f2),v in ig.items():
        ig_feat[f1] += max(v,0)
        ig_feat[f2] += max(v,0)


    # ---------------------------------------------------
    # 5) Variance Inflation Factors
    # ---------------------------------------------------
    print("\nComputing VIF...")
    vif = pd.Series(
        [variance_inflation_factor(Z.values,i)
         for i in range(Z.shape[1])],
        index=Z.columns
    )


    # ---------------------------------------------------
    # 6) Build unified scoring table
    # ---------------------------------------------------
    report = stats.set_index("feature")
    report["MI"] = mi
    report["IG"] = ig_feat
    report["VIF"] = vif

    # ----- NEW: normalize metrics to 0–1 before combining -----
    for col in ["MI", "auc", "cohen_d", "IG"]:
        vals = report[col].values
        # for IG, ignore negatives (no "penalty" for low interaction)
        if col == "IG":
            vals = np.maximum(vals, 0.0)
            report[col] = vals

        vmin = report[col].min()
        vmax = report[col].max()
        if vmax > vmin:
            report[f"{col}_norm"] = (report[col] - vmin) / (vmax - vmin)
        else:
            # all equal → just set to 0.5
            report[f"{col}_norm"] = 0.5

    # Equal weights: 0.25 each after normalization

    w_mi, w_auc, w_d, w_ig = weights

    report["rank_score"] = (
            w_mi * report["MI_norm"]
            + w_auc * report["auc_norm"]
            + w_d * report["cohen_d_norm"]
            + w_ig * report["IG_norm"]
    )

    # Sort by new rank_score
    report = report.sort_values("rank_score", ascending=False)

    print("\n===== FEATURE SCORE TABLE =====\n")
    print(report)


    # ---------------------------------------------------
    # 7) Correlation matrix
    # ---------------------------------------------------
    corr = Z.corr().abs()

    # ---------------------------------------------------
    # 8) Forward greedy selection (VIF FIRST, then CORR)
    # ---------------------------------------------------
    print("\nSelecting features...")

    selected = []

    for f in report.index:

        # ---- Always accept first feature ----
        if len(selected) == 0:
            print(f"✓ Selecting FIRST feature : {f}")
            selected.append(f)
            continue

        # ---------------------------------------------------
        # 1) CONDITIONAL VIF CHECK (FIRST)
        # ---------------------------------------------------
        candidate_set = selected + [f]
        Z_sub = Z[candidate_set]
        vif_sub = compute_vif_subset(Z_sub)
        vif_val = vif_sub[f]

        if len(selected) == 1:
            if vif_val > vif_soft:
                print(f"X Reject {f} — VIF={vif_val:.2f} > VIF_soft={vif_soft}")
                continue
        else:
            if vif_val > vif_hard:
                print(f"X Reject {f} — VIF={vif_val:.2f} > VIF_hard={vif_hard}")
                continue

        # ---------------------------------------------------
        # 2) CORRELATION CHECK (SECOND)
        # ---------------------------------------------------
        high_corr = False
        for s in selected:
            r = corr.loc[f, s]
            if r > corr_thresh:
                print(
                    f"X Reject {f} — corr({f},{s})={r:.3f} > {corr_thresh}"
                )
                high_corr = True
                break

        if high_corr:
            continue

        # ---------------------------------------------------
        # 3) ACCEPT FEATURE
        # ---------------------------------------------------
        print(
            f"✓ Accept {f} — VIF={vif_val:.2f}, "
            f"max_corr={max(corr.loc[f, s] for s in selected):.3f}"
        )
        selected.append(f)

        if len(selected) >= top_k:
            break

    # ---------------------------------------------------
    # 9) Summary
    # ---------------------------------------------------
    print("\n=========== SUMMARY ===========")
    print(f"Total candidates : {len(X_df.columns)}")
    print(f"Selected         : {selected}")
    print("================================\n")


    # ---------------------------------------------------
    # 10) Plot feature-wise AUC
    # ---------------------------------------------------
    if verbose:
        auc_series = report.loc[selected,"auc"].sort_values()

        plt.figure(figsize=(7,5))
        auc_series.plot(kind="barh")
        plt.title("Univariate AUC of Selected Features")
        plt.xlabel("AUC")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------
    # OPTIONAL QUIET MODE & LIGHTWEIGHT RETURN
    # ---------------------------------------------------
    # During weight-grid search, we do NOT want:
    #   - plots
    #   - prints
    #   - large objects (corr, vif, full report)
    #
    # We only need the selected feature list.
    # This avoids massive I/O and speeds up search.
    # ---------------------------------------------------

    if return_selected_only:
        return selected

    return selected, report, corr, vif




# ---------------------------------------------------------------
#   2-VARIABLE VISUALISATION
# ---------------------------------------------------------------
def plot_feature_pair(df_all, feature_x, feature_y, label_col="Label_main"):
    """
    2-D scatter plot of two features.
    Useful to visually inspect separation.
    """

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df_all,
        x=feature_x,
        y=feature_y,
        hue=label_col,
        s=70,
        alpha=0.85
    )

    plt.title(f"{feature_x}  vs  {feature_y}")
    plt.legend(title="Class")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
#   3-VARIABLE VISUALISATION
# ---------------------------------------------------------------


def plot_feature_triplet(df_all, f1, f2, f3, label_col="Label_main"):
    """
    3-D scatter plot of three features.
    Helps visualise curved / nonlinear separability.
    """

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        df_all[f1], df_all[f2], df_all[f3],
        c=df_all[label_col],
        cmap="coolwarm",
        s=55,
        alpha=0.9
    )

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3)
    ax.set_title(f"3-D feature clustering: {f1}, {f2}, {f3}")

    legend = ax.legend(*scatter.legend_elements(), title="Class")
    ax.add_artist(legend)

    plt.tight_layout()
    plt.show()


# def compute_vif_subset(Z_sub):
#     """
#     Compute VIF only on the selected feature subset.
#     """
#     return pd.Series(
#         [variance_inflation_factor(Z_sub.values, i)
#          for i in range(Z_sub.shape[1])],
#         index=Z_sub.columns
#     )
