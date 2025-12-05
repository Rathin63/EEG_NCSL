"""
HC vs ADHD classification + ADHD subclass classification

Primary task  : Binary classification (HC vs ADHD)
Secondary task: Multi-class classification (ADHD_Low vs ADHD_Med vs ADHD_High)

Classifiers:
- Logistic Regression
- Random Forest
- SVM (RBF kernel)
- k-Nearest Neighbors
- Gaussian Naive Bayes
- MLP Neural Network (shallow feed-forward)

Outputs:
- Cross-validation metrics (CSV) for both tasks
- ROC curves (PNG) for HC vs ADHD
- Confusion matrices + classification reports (TXT)
- Simple feature-importance file for Random Forest & LogReg

Requirements:
- numpy, pandas, matplotlib
- scikit-learn

Author: (your name / date)
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
)

# -------------------------------------------------------------------
# USER SETTINGS  (EDIT THIS BLOCK)
# -------------------------------------------------------------------

# Excel files: 4 groups
file_HC        = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\HC\Results\BatchSummary_HC.xlsx"
file_ADHD_Low  = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Low\Results\BatchSummary_ADHD_Low.xlsx"
file_ADHD_Med  = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_Med\Results\BatchSummary_ADHD_Med.xlsx"
file_ADHD_High = r"E:\JHU_Postdoc\Research\TDBrain\TD_BRAIN_code\BRAIN_code\Sample\diff_data2\ADHD\ADHD_High\Results\BatchSummary_ADHD_High.xlsx"

# EXACT feature names as they appear in Excel
# (Example – change to match your BatchSummary headers)
FEATURES = [
    "TBR",
    "CPO Sink (Good)",
    "AI_FT_Good",
    "Sink Gradient (Good)",
    "FT/CPO Sink (Good)"

]

# Output folder
BASE_DIR = os.path.dirname(file_HC)
OUT_DIR = os.path.join(BASE_DIR, "ADHD_Classification_4Groups")
os.makedirs(OUT_DIR, exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

# Number of CV folds (for both tasks)
N_FOLDS_PRIMARY = 5   # HC vs ADHD
N_FOLDS_SECONDARY = 5 # ADHD_Low vs ADHD_Med vs ADHD_High


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_group(path, label_main, label_subclass):
    """
    Load one group Excel file, attach:
      - Label_main    : 0 = HC, 1 = ADHD
      - Label_subclass: "HC", "Low", "Med", "High"
    """
    df = pd.read_excel(path)
    df["Label_main"] = label_main
    df["Label_subclass"] = label_subclass
    return df


print("Loading data from Excel...")
df_HC        = load_group(file_HC,        label_main=0, label_subclass="HC")
df_ADHD_Low  = load_group(file_ADHD_Low,  label_main=1, label_subclass="Low")
df_ADHD_Med  = load_group(file_ADHD_Med,  label_main=1, label_subclass="Med")
df_ADHD_High = load_group(file_ADHD_High, label_main=1, label_subclass="High")

df_all = pd.concat([df_HC, df_ADHD_Low, df_ADHD_Med, df_ADHD_High],
                   ignore_index=True)

print(f"Total subjects: {len(df_all)} "
      f"(HC = {len(df_HC)}, ADHD = {len(df_all) - len(df_HC)})")

# Check feature presence
missing = [f for f in FEATURES if f not in df_all.columns]
if missing:
    raise ValueError(f"Missing feature columns in Excel: {missing}")

# Drop rows with NaNs in selected features
df_all = df_all.dropna(subset=FEATURES).reset_index(drop=True)
print(f"After dropping NaNs in features, N = {len(df_all)}")

# -------------------------------------------------------------------
# PREPARE LABELS FOR BOTH TASKS
# -------------------------------------------------------------------

# Primary task: HC (0) vs ADHD (1)
X_primary = df_all[FEATURES].astype(float).values
y_primary = df_all["Label_main"].values

# Load data
# Prepare X_primary, y_primary
# -------------- INSERT TUNING HERE --------------
# Define models dictionary
# CV evaluation

# ============================================================
# HYPERPARAMETER TUNING (SVM-RBF, RandomForest, MLP-NN)
# ============================================================

from sklearn.model_selection import GridSearchCV, StratifiedKFold

print("\n=== Running Hyperparameter Tuning (GridSearchCV) ===")

# Common CV splitter
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ----------------------------------------
# 1. SVM-RBF Hyperparameter Grid
# ----------------------------------------
svm_param_grid = {
    "clf__C": [0.1, 1, 5, 10, 20],          # SVM soft margin — increase = less regularization
    "clf__gamma": ["scale", "auto", 0.1, 0.01, 0.001],   # RBF kernel width
}

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True))
])

grid_svm = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=inner_cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nTuning SVM-RBF...")
grid_svm.fit(X_primary, y_primary)
print("Best SVM Params:", grid_svm.best_params_)
print("Best SVM AUC:", grid_svm.best_score_)

# ----------------------------------------
# 2. RandomForest Hyperparameter Grid
# ----------------------------------------
rf_param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
}

rf_clf = RandomForestClassifier(random_state=RANDOM_STATE)

grid_rf = GridSearchCV(
    estimator=rf_clf,
    param_grid=rf_param_grid,
    cv=inner_cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nTuning RandomForest...")
grid_rf.fit(X_primary, y_primary)
print("Best RF Params:", grid_rf.best_params_)
print("Best RF AUC:", grid_rf.best_score_)

# ----------------------------------------
# 3. MLP Neural Network Hyperparameter Grid
# ----------------------------------------
mlp_param_grid = {
    "clf__hidden_layer_sizes": [(32,), (64,), (32,16), (64,32)],
    "clf__alpha": [1e-4, 1e-3, 1e-2],                # L2 regularization
    "clf__learning_rate_init": [1e-3, 5e-3, 1e-2],
    "clf__activation": ["relu", "tanh"]              # activation functions
}

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(
        max_iter=2000,          # Increased iterations
        early_stopping=True,    # Stop when validation stops improving
        n_iter_no_change=20,    # Patience
        random_state=RANDOM_STATE
    ))
])


grid_mlp = GridSearchCV(
    estimator=mlp_pipeline,
    param_grid=mlp_param_grid,
    cv=inner_cv,
    scoring="roc_auc",
    n_jobs=-1
)

print("\nTuning MLP Neural Network...")
grid_mlp.fit(X_primary, y_primary)
print("Best MLP Params:", grid_mlp.best_params_)
print("Best MLP AUC:", grid_mlp.best_score_)

print("\n=== Hyperparameter Tuning Completed ===")


# Secondary task: ADHD subclass only
df_adhd_only = df_all[df_all["Label_main"] == 1].copy()
subclass_map = {"Low": 0, "Med": 1, "High": 2}  # internal encoding
df_adhd_only["Subclass_int"] = df_adhd_only["Label_subclass"].map(subclass_map)

X_secondary = df_adhd_only[FEATURES].astype(float).values
y_secondary = df_adhd_only["Subclass_int"].values

print(f"ADHD-only subjects for subclass task: {len(df_adhd_only)} "
      f"(Low={sum(y_secondary==0)}, Med={sum(y_secondary==1)}, High={sum(y_secondary==2)})")


# -------------------------------------------------------------------
# DEFINE CLASSIFIERS
# (Hyperparameters marked with 'TUNE HERE')
# -------------------------------------------------------------------

models = {
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,          # TUNE HERE (inverse regularization strength)
            penalty="l2",
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,    # TUNE HERE (number of trees)
        max_depth=None,      # TUNE HERE (tree depth; None = full)
        min_samples_split=2, # TUNE HERE
        min_samples_leaf=1,  # TUNE HERE
        random_state=RANDOM_STATE
    ),

    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=1.0,          # TUNE HERE (margin softness)
            gamma="scale",  # TUNE HERE (RBF width)
            probability=True,
            random_state=RANDOM_STATE
        ))
    ]),

    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=5,  # TUNE HERE (k)
            weights="distance"  # TUNE HERE ("uniform" or "distance")
        ))
    ]),

    "NaiveBayes": Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", GaussianNB())
    ]),

    "MLP_NN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(16, 8),  # Smaller network
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=0.005,
            early_stopping=True,
            n_iter_no_change=20,
            max_iter=2000,
            random_state=RANDOM_STATE
        ))
    ])
}


# -------------------------------------------------------------------
# HELPER: GENERIC CV EVALUATION
# -------------------------------------------------------------------

def evaluate_models_cv(X, y, models_dict, n_folds, task_name, is_multiclass=False):
    """
    Run StratifiedKFold cross-validation on all models.

    For binary:
        - Accuracy, F1, AUC

    For multiclass:
        - Accuracy, macro-F1, macro-AUC (OvR)

    Returns:
        cv_df: DataFrame with per-model CV summary.
    """
    print(f"\n=== {task_name}: {n_folds}-fold Stratified CV ===")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    results = []

    for name, model in models_dict.items():
        accs, f1s, aucs = [], [], []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            # Accuracy
            acc = accuracy_score(y_te, y_pred)
            accs.append(acc)

            # F1
            if is_multiclass:
                f1 = f1_score(y_te, y_pred, average="macro")
            else:
                f1 = f1_score(y_te, y_pred)
            f1s.append(f1)

            # AUC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_te)
                if is_multiclass:
                    # macro-averaged OvR AUC
                    auc = roc_auc_score(y_te, y_score, multi_class="ovr",
                                        average="macro")
                else:
                    auc = roc_auc_score(y_te, y_score[:, 1])
            else:
                # Fall back if no predict_proba (rare in this setup)
                if not is_multiclass:
                    y_score = model.decision_function(X_te)
                    auc = roc_auc_score(y_te, y_score)
                else:
                    auc = np.nan  # Not well-defined without probabilities
            aucs.append(auc)

        results.append({
            "Model": name,
            "CV_Accuracy_Mean": np.mean(accs),
            "CV_Accuracy_SD": np.std(accs),
            "CV_F1_Mean": np.mean(f1s),
            "CV_F1_SD": np.std(f1s),
            "CV_AUC_Mean": np.mean(aucs),
            "CV_AUC_SD": np.std(aucs),
        })

        print(f"{name}: "
              f"ACC = {np.mean(accs):.3f} ± {np.std(accs):.3f}, "
              f"F1 = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}, "
              f"AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    cv_df = pd.DataFrame(results).sort_values("CV_AUC_Mean", ascending=False)
    return cv_df


# -------------------------------------------------------------------
# PRIMARY TASK: HC vs ADHD
# -------------------------------------------------------------------

cv_primary = evaluate_models_cv(
    X_primary, y_primary,
    models_dict=models,
    n_folds=N_FOLDS_PRIMARY,
    task_name="Primary (HC vs ADHD)",
    is_multiclass=False
)

primary_csv = os.path.join(OUT_DIR, "cv_primary_HC_vs_ADHD.csv")
cv_primary.to_csv(primary_csv, index=False)
print(f"\nPrimary CV results saved to: {primary_csv}")

# Identify best model by AUC for primary task
best_model_name_primary = cv_primary.iloc[0]["Model"]
print(f"\nBest primary model by AUC: {best_model_name_primary}")
best_model_primary = models[best_model_name_primary]


# -------------------------------------------------------------------
# SECONDARY TASK: ADHD_Low vs ADHD_Med vs ADHD_High
# -------------------------------------------------------------------

cv_secondary = evaluate_models_cv(
    X_secondary, y_secondary,
    models_dict=models,
    n_folds=N_FOLDS_SECONDARY,
    task_name="Secondary (ADHD subclass: Low/Med/High)",
    is_multiclass=True
)

secondary_csv = os.path.join(OUT_DIR, "cv_secondary_ADHD_subclass.csv")
cv_secondary.to_csv(secondary_csv, index=False)
print(f"\nSecondary CV results saved to: {secondary_csv}")

best_model_name_secondary = cv_secondary.iloc[0]["Model"]
print(f"\nBest secondary model by AUC: {best_model_name_secondary}")
best_model_secondary = models[best_model_name_secondary]


# -------------------------------------------------------------------
# TRAIN/TEST SPLIT FOR PRIMARY TASK + ROC CURVES
# -------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_primary, y_primary,
    test_size=0.3, stratify=y_primary, random_state=RANDOM_STATE
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
roc_ax.set_title("ROC Curves (HC vs ADHD, 30% Test Split)")
roc_ax.legend(loc="lower right")

roc_png = os.path.join(OUT_DIR, "roc_curves_primary.png")
plt.tight_layout()
plt.savefig(roc_png, dpi=200)
plt.close(roc_fig)
print(f"\nROC curves (primary) saved to: {roc_png}")


# -------------------------------------------------------------------
# CONFUSION MATRIX + REPORT FOR BEST PRIMARY MODEL
# -------------------------------------------------------------------

best_model_primary.fit(X_train, y_train)
y_pred_primary = best_model_primary.predict(X_test)

cm_primary = confusion_matrix(y_test, y_pred_primary)
report_primary = classification_report(
    y_test, y_pred_primary, target_names=["HC", "ADHD"]
)

cm_primary_path = os.path.join(OUT_DIR, "primary_confusion_and_report.txt")
with open(cm_primary_path, "w") as f:
    f.write(f"Best model (primary): {best_model_name_primary}\n\n")
    f.write("Confusion Matrix (HC vs ADHD):\n")
    f.write(str(cm_primary) + "\n\n")
    f.write("Classification report:\n")
    f.write(report_primary)

print(f"Primary confusion matrix + report saved to: {cm_primary_path}")


# -------------------------------------------------------------------
# CONFUSION MATRIX + REPORT FOR BEST SECONDARY MODEL
# -------------------------------------------------------------------

# Train/test split within ADHD subjects only
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_secondary, y_secondary,
    test_size=0.3, stratify=y_secondary, random_state=RANDOM_STATE
)

best_model_secondary.fit(X_train_s, y_train_s)
y_pred_secondary = best_model_secondary.predict(X_test_s)

cm_secondary = confusion_matrix(y_test_s, y_pred_secondary)
target_names_sub = ["Low", "Med", "High"]
report_secondary = classification_report(
    y_test_s, y_pred_secondary, target_names=target_names_sub
)

cm_secondary_path = os.path.join(OUT_DIR, "secondary_confusion_and_report.txt")
with open(cm_secondary_path, "w") as f:
    f.write(f"Best model (secondary): {best_model_name_secondary}\n\n")
    f.write("Confusion Matrix (ADHD subclass: Low/Med/High):\n")
    f.write(str(cm_secondary) + "\n\n")
    f.write("Classification report:\n")
    f.write(report_secondary)

print(f"Secondary confusion matrix + report saved to: {cm_secondary_path}")


# -------------------------------------------------------------------
# SIMPLE FEATURE IMPORTANCE (RF & LogReg) – OPTIONAL
# -------------------------------------------------------------------

# Random Forest importance (primary)
if "RandomForest" in models:
    rf = models["RandomForest"]
    rf.fit(X_primary, y_primary)
    rf_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)

    rf_path = os.path.join(OUT_DIR, "feature_importance_RandomForest.csv")
    rf_importance.to_csv(rf_path, index=False)
    print(f"Random Forest feature importance saved to: {rf_path}")

# Logistic Regression coefficients (primary)
if "LogReg" in models:
    lr = models["LogReg"]
    lr.fit(X_primary, y_primary)
    clf_lr = lr.named_steps["clf"]
    coefs = clf_lr.coef_[0]

    lr_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefs
    }).sort_values("Coefficient", ascending=False)

    lr_path = os.path.join(OUT_DIR, "feature_importance_LogReg.csv")
    lr_importance.to_csv(lr_path, index=False)
    print(f"LogReg coefficients saved to: {lr_path}")

print("\nDone. Check the output folder for CSVs, plots, and reports:")
print(OUT_DIR)
