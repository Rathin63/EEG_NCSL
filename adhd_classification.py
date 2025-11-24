"""
ADHD vs HC classification using top 3 EEG features:

- TBR
- FT/CPO Sink (Good)
- AI_FT_Good

Runs 5-fold stratified CV for Logistic Regression, Random Forest, and SVM-RBF,
then trains each model on full data and generates ROC curves and plots.

Place this file anywhere (e.g., preprocessing/utils/) and run it directly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# -------------------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------------------

file_A = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\BatchOps\BatchSummary_HC.xlsx"
file_B = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\BatchOps\BatchSummary_ADHD.xlsx"

# Three best features (exact column names in Excel)
FEATURES = ["TBR", "FT Sink (Good)", "CPO Sink (Good)", "AI_FT_All", "AI_CPO_All"]

# Output directory
BASE_DIR = os.path.dirname(file_A)
OUT_DIR = os.path.join(BASE_DIR, "ADHD_Classification")
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_group(path, label):
    """Load one group and attach label column."""
    df = pd.read_excel(path)
    df["Label"] = label
    return df


print("Loading data...")
df_hc = load_group(file_A, label=0)     # Healthy controls
df_adhd = load_group(file_B, label=1)   # ADHD

df_all = pd.concat([df_hc, df_adhd], ignore_index=True)

print(f"Total subjects: {len(df_all)} "
      f"(HC = {len(df_hc)}, ADHD = {len(df_adhd)})")

# Check that required feature columns exist
missing = [f for f in FEATURES if f not in df_all.columns]
if missing:
    raise ValueError(f"Missing columns in Excel: {missing}")

X = df_all[FEATURES].astype(float).values
y = df_all["Label"].values


# -------------------------------------------------------------------
# DEFINE MODELS
# -------------------------------------------------------------------

models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42
    ),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True))
    ]),
}


# -------------------------------------------------------------------
# 5-FOLD STRATIFIED CROSS-VALIDATION
# -------------------------------------------------------------------

print("\nRunning 5-fold stratified cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name, model in models.items():
    accs, aucs = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # Accuracy
        acc = accuracy_score(y_te, y_pred)
        accs.append(acc)

        # AUC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        else:
            # For some models, decision_function is available
            y_score = model.decision_function(X_te)
        auc = roc_auc_score(y_te, y_score)
        aucs.append(auc)

    cv_results.append({
        "Model": name,
        "CV_Accuracy_Mean": np.mean(accs),
        "CV_Accuracy_SD": np.std(accs),
        "CV_AUC_Mean": np.mean(aucs),
        "CV_AUC_SD": np.std(aucs),
    })

    print(f"{name}: "
          f"ACC = {np.mean(accs):.3f} ± {np.std(accs):.3f}, "
          f"AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")


cv_df = pd.DataFrame(cv_results)
cv_path = os.path.join(OUT_DIR, "classification_results.csv")
cv_df.to_csv(cv_path, index=False)
print(f"\nCV metrics saved to: {cv_path}")


# -------------------------------------------------------------------
# TRAIN MODELS ON FULL DATA FOR PLOTTING & INTERPRETATION
# -------------------------------------------------------------------

# Train-test split for ROC visualization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

roc_fig, roc_ax = plt.subplots(figsize=(7, 6))

for name, model in models.items():
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc_val = roc_auc_score(y_test, y_score)

    roc_ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")

# Chance line
roc_ax.plot([0, 1], [0, 1], "k--", label="Chance")

roc_ax.set_xlabel("False Positive Rate")
roc_ax.set_ylabel("True Positive Rate")
roc_ax.set_title("ROC Curves (30% Test Split)")
roc_ax.legend(loc="lower right")

roc_png = os.path.join(OUT_DIR, "roc_curves.png")
plt.tight_layout()
plt.savefig(roc_png, dpi=200)
plt.close(roc_fig)
print(f"ROC curves saved to: {roc_png}")


# -------------------------------------------------------------------
# LOGISTIC REGRESSION FEATURE IMPORTANCE
# -------------------------------------------------------------------

# Refit LogReg on full data for interpretation
logreg = models["LogReg"]
logreg.fit(X, y)

clf_lr = logreg.named_steps["clf"]
scaler = logreg.named_steps["scaler"]

coefs = clf_lr.coef_[0]
# Use scaled feature names as-is; scaling already handled in pipeline
feat_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": coefs
}).sort_values(by="Coefficient", ascending=False)

fi_path = os.path.join(OUT_DIR, "feature_importance_logreg.txt")
with open(fi_path, "w") as f:
    f.write("Logistic Regression coefficients (after scaling):\n\n")
    for _, row in feat_importance.iterrows():
        f.write(f"{row['Feature']}: {row['Coefficient']:.4f}\n")

print(f"LogReg feature importance saved to: {fi_path}")


# -------------------------------------------------------------------
# SCATTER PLOT: TBR vs AI_FT_Good
# -------------------------------------------------------------------

feature_x = "TBR"
feature_y = "AI_FT_All"

x_vals = df_all[feature_x].astype(float).values
y_vals = df_all[feature_y].astype(float).values

plt.figure(figsize=(7, 6))
for label, marker, color, name in [(0, "o", "tab:blue", "HC"),
                                   (1, "s", "tab:orange", "ADHD")]:
    mask = (df_all["Label"] == label)
    plt.scatter(x_vals[mask], y_vals[mask], marker=marker,
                alpha=0.8, label=name)

plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title("TBR vs AI_FT_All")
plt.legend()
plt.grid(True, alpha=0.3)

scatter_path = os.path.join(OUT_DIR, "scatter_TBR_vs_AI_FT_All.png")
plt.tight_layout()
plt.savefig(scatter_path, dpi=200)
plt.close()
print(f"Scatter plot saved to: {scatter_path}")


# -------------------------------------------------------------------
# CONFUSION MATRIX & REPORT (BEST MODEL = LOGREG BY DEFAULT)
# -------------------------------------------------------------------

y_pred = logreg.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["HC", "ADHD"])

cm_path = os.path.join(OUT_DIR, "confusion_matrix_and_report.txt")
with open(cm_path, "w") as f:
    f.write("Confusion Matrix (LogReg on 30% test split):\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification report:\n")
    f.write(report)

print(f"Confusion matrix and classification report saved to: {cm_path}")

print("\nDone. Check the ADHD_Classification folder for outputs.\n")
