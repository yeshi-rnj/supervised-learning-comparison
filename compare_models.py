"""
Comparative study of 5 supervised learning algorithms on the
Pima Indians Diabetes dataset (binary classification).

Models: Logistic Regression, Decision Tree, K-Nearest Neighbors,
        Support Vector Machine (RBF), Gaussian Naive Bayes

Run:  python compare_models.py
Outputs:
    - results/comparison_table.csv
    - results/comparison_chart.png
    - results/confusion_matrices.png
    - results/roc_curves.png
    - prints a summary to stdout
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42
os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
df = pd.read_csv("diabetes.csv", names=COLS)

print(f"Dataset shape: {df.shape}")
print(f"Class balance:\n{df['Outcome'].value_counts(normalize=True)}\n")

# ---------------------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------------------
# In this dataset, 0 is not a physiologically valid value for these
# columns -- it actually encodes a missing measurement. We treat
# those zeros as missing values and impute with the column median
# (computed on the TRAIN split only, to avoid leakage -- see below).
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in ZERO_AS_MISSING:
    df[col] = df[col].replace(0, np.nan)

print("Missing values introduced by cleaning:")
print(df[ZERO_AS_MISSING].isna().sum(), "\n")

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# No categorical columns in this dataset, so no encoding step is
# needed here -- but if there were, we'd one-hot encode them at this
# point, fit on train only, same as the imputer/scaler below.

# ---------------------------------------------------------------------
# 3. Single train/test split shared by every model
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Impute missing values using TRAIN medians only, then apply to both
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# ---------------------------------------------------------------------
# 4. Scaling (fit on train only, applied to both)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=X_test.columns, index=X_test.index
)

# ---------------------------------------------------------------------
# 5. Define models
# ---------------------------------------------------------------------
# Tree-based models don't need scaling (split thresholds are scale
# invariant); distance/margin-based models (KNN, SVM) and gradient
# based linear models (LogReg) do. Naive Bayes' Gaussian likelihood
# is technically scale-invariant in effect, but we keep it on raw
# features here since it models per-feature distributions directly
# and scaling doesn't change its decision boundary.
models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "scaled": True,
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "scaled": False,
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(n_neighbors=15),
        "scaled": True,
    },
    "Support Vector Machine": {
        "model": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "scaled": True,
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "scaled": False,
    },
}

# ---------------------------------------------------------------------
# 6. Train + evaluate every model on the identical split
# ---------------------------------------------------------------------
results = []
roc_data = {}
cm_data = {}

for name, cfg in models.items():
    m = cfg["model"]
    if cfg["scaled"]:
        Xtr, Xte = X_train_scaled, X_test_scaled
    else:
        Xtr, Xte = X_train, X_test

    m.fit(Xtr, y_train)
    y_pred = m.predict(Xte)
    y_proba = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": auc,
    })

    cm_data[name] = confusion_matrix(y_test, y_pred)
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# ---------------------------------------------------------------------
# 7. Comparison table
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
results_df_display = results_df.copy()
for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
    results_df_display[c] = results_df_display[c].round(4)

results_df_display.to_csv("results/comparison_table.csv", index=False)
print("\n\n===== FINAL COMPARISON TABLE (ranked by F1 Score) =====")
print(results_df_display.to_string(index=False))

# ---------------------------------------------------------------------
# 8. Comparison bar chart
# ---------------------------------------------------------------------
sns.set_theme(style="whitegrid")
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
plot_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot,
                           var_name="Metric", value_name="Score")

plt.figure(figsize=(11, 6))
ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
plt.title("Model Comparison on Pima Indians Diabetes Dataset", fontsize=14, weight="bold")
plt.ylim(0, 1.0)
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("results/comparison_chart.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 9. Confusion matrices
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, (name, cm) in zip(axes, cm_data.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Diab.", "Diab."], yticklabels=["No Diab.", "Diab."])
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------
# 10. ROC curves
# ---------------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("results/roc_curves.png", dpi=150)
plt.close()

print("\nSaved: results/comparison_table.csv, comparison_chart.png, confusion_matrices.png, roc_curves.png")
