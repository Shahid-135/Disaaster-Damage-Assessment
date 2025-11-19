
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    cohen_kappa_score,
)
from sklearn.preprocessing import label_binarize

# -------------------------
# Inputs (replace with your arrays)
# -------------------------
# test_labels = ...
# y_prediction = ...  # either 1D predicted labels OR 2D array of scores/probabilities

# -------------------------
# Configuration
# -------------------------
save_dir = "results2_final"
os.makedirs(save_dir, exist_ok=True)
sns.set_style("whitegrid")
fig_fname = "combined_metrics_grid.png"
dpi = 300

# -------------------------
# Sanity convert to numpy
# -------------------------
test_labels = np.asarray(test_labels)
y_prediction = np.asarray(y_prediction)

# -------------------------
# Determine whether y_prediction contains scores or discrete labels
# -------------------------
if y_prediction.ndim == 2:
    # treat as probability / score matrix
    y_scores = y_prediction
else:
    y_scores = None

# -------------------------
# Infer classes and create binarized matrices
# -------------------------
if y_scores is None:
    unique_vals = np.unique(np.concatenate([test_labels, y_prediction]))
else:
    unique_vals = np.unique(test_labels)

classes = list(unique_vals)
n_classes = len(classes)
class_names = [str(c) for c in classes]

# Binarize true labels aligned to classes
y_true_bin = label_binarize(test_labels, classes=classes)
if y_true_bin.ndim == 1:
    y_true_bin = y_true_bin.reshape(-1, 1)

# Build / validate score matrix and predicted labels
if y_scores is None:
    y_pred = np.asarray(y_prediction)
    missing = set(np.unique(y_pred)) - set(classes)
    if missing:
        raise ValueError(f"Predicted labels contain unseen values: {missing}")
    y_pred_indices = np.array([classes.index(v) for v in y_pred])
    y_scores = np.zeros((len(y_pred), n_classes), dtype=float)
    y_scores[np.arange(len(y_pred)), y_pred_indices] = 1.0
else:
    if y_scores.shape[1] != n_classes:
        raise ValueError(
            f"y_score has {y_scores.shape[1]} columns but detected {n_classes} classes. "
            "Ensure columns of y_score correspond to `classes` (in the order of `classes`)."
        )
    y_pred_indices = np.argmax(y_scores, axis=1)
    y_pred = np.array([classes[idx] for idx in y_pred_indices])

# -------------------------
# Classic discrete metrics and classification report
# -------------------------
report = classification_report(test_labels, y_pred, labels=classes, output_dict=True, zero_division=0)
per_class_df = pd.DataFrame(report).transpose()

# Confusion matrix (rows=true, cols=pred)
cm = confusion_matrix(test_labels, y_pred, labels=classes)

# Compute per-class IoU from confusion matrix
tps = np.diag(cm).astype(float)
fps = cm.sum(axis=0).astype(float) - tps
fns = cm.sum(axis=1).astype(float) - tps
denom = tps + fps + fns
iou_per = np.divide(tps, denom, out=np.zeros_like(tps), where=denom != 0)
mean_iou = float(np.mean(iou_per)) if len(iou_per) > 0 else 0.0

# Add IoU and confusion-derived counts to per_class_df (if present)
# Ensure index alignment: per_class_df rows for classes then 'macro avg', 'weighted avg', 'accuracy'
for idx, cls in enumerate(classes):
    if cls in per_class_df.index:
        per_class_df.loc[cls, "iou"] = float(iou_per[idx])
        per_class_df.loc[cls, "tp"] = int(tps[idx])
        per_class_df.loc[cls, "fp"] = int(fps[idx])
        per_class_df.loc[cls, "fn"] = int(fns[idx])
    else:
        # add new row if not present (defensive)
        per_class_df.loc[cls] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1-score": 0.0,
            "support": int(0),
            "iou": float(iou_per[idx]),
            "tp": int(tps[idx]),
            "fp": int(fps[idx]),
            "fn": int(fns[idx]),
        }

# Reorder columns for readability if they exist
cols_order = ["precision", "recall", "f1-score", "support", "iou", "tp", "fp", "fn"]
existing_cols = [c for c in cols_order if c in per_class_df.columns]
per_class_df = per_class_df[existing_cols]

# -------------------------
# Plot: Confusion Matrix, PR curves, ROC curves
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=dpi)

# 1) Confusion Matrix
ax = axes[0]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False, annot_kws={"size": 10})
ax.set_title("Confusion Matrix", fontsize=14)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_xticks(np.arange(len(class_names)) + 0.5)
ax.set_yticks(np.arange(len(class_names)) + 0.5)
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names, rotation=0)

# 2) Precision-Recall Curves
ax = axes[1]
for i in range(n_classes):
    if y_true_bin.shape[1] <= i or y_true_bin[:, i].sum() == 0:
        continue
    try:
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        ax.step(recall, precision, where="post", label=f"{class_names[i]}")
    except ValueError:
        continue
ax.set_title("Precision-Recall Curves", fontsize=14)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True)

# 3) ROC Curves
ax = axes[2]
for i in range(n_classes):
    if y_true_bin.shape[1] <= i or y_true_bin[:, i].sum() == 0 or (y_true_bin.shape[0] - y_true_bin[:, i].sum()) == 0:
        continue
    try:
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    except ValueError:
        continue
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
ax.set_title("ROC Curves", fontsize=14)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, fig_fname))
plt.close()

# -------------------------
# Save metrics (summary + per-class)
# -------------------------
metrics = {
    "accuracy": float(accuracy_score(test_labels, y_pred)),
    # macro aggregates
    "precision_macro": float(precision_score(test_labels, y_pred, average="macro", zero_division=0)),
    "recall_macro": float(recall_score(test_labels, y_pred, average="macro", zero_division=0)),
    "f1_macro": float(f1_score(test_labels, y_pred, average="macro", zero_division=0)),
    # weighted aggregates
    "precision_weighted": float(precision_score(test_labels, y_pred, average="weighted", zero_division=0)),
    "recall_weighted": float(recall_score(test_labels, y_pred, average="weighted", zero_division=0)),
    "f1_weighted": float(f1_score(test_labels, y_pred, average="weighted", zero_division=0)),
    # micro aggregates
    "precision_micro": float(precision_score(test_labels, y_pred, average="micro", zero_division=0)),
    "recall_micro": float(recall_score(test_labels, y_pred, average="micro", zero_division=0)),
    "f1_micro": float(f1_score(test_labels, y_pred, average="micro", zero_division=0)),
    # additional metrics
    "cohen_kappa": float(cohen_kappa_score(test_labels, y_pred)),
    "n_samples": int(len(test_labels)),
    "classes": class_names,
}

with open(os.path.join(save_dir, "local_metrics.json"), "w", encoding="utf8") as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)

per_class_df.to_csv(os.path.join(save_dir, "per_class_metrics.csv"))

# also save confusion matrix for reference (numpy)
np.save(os.path.join(save_dir, "confusion_matrix.npy"), cm)

print(f"Saved combined figure and metrics to '{save_dir}'.")
print("Wrote: combined figure, local_metrics.json, per_class_metrics.csv, confusion_matrix.npy")
