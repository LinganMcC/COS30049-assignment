import joblib
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Load model and data 
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

feature_cols = [
    "n_words", "long_word_ratio", "func_word_ratio", "unique_word_ratio",
    "comma_rate", "exclaim", "apostrophe_rate", "starts_with_transition",
    "rel_len_deviation", "noun_ratio", "adjective_ratio", "adverb_ratio",
    "pronoun_ratio", "auxiliary_ratio", "conjunction_ratio",
]

train_df = pd.read_csv("data/features/drcat_train_final_sentence_features.csv")
test_df = pd.read_csv("data/features/drcat_test_final_sentence_features.csv")

# Check for train/test leakage
print("Shared document_ids:", len(set(train_df["document_id"]) & set(test_df["document_id"])))
print("Shared group_keys:  ", len(set(train_df["group_key"]) & set(test_df["group_key"])))

#  Map clusters to labels
train_clusters = kmeans.predict(scaler.transform(train_df[feature_cols]))

print("\nCluster composition (training):")
print(pd.crosstab(train_clusters, train_df["label"],
                  rownames=["cluster"], colnames=["true label"]))
print(pd.crosstab(train_clusters, train_df["label"],
                  rownames=["cluster"], colnames=["true label"],
                  normalize="index"))

# The cluster with the highest share of AI sentences -> label 1, the other -> label 0
ai_share = train_df.groupby(train_clusters)["label"].mean()
cluster_to_label = {c: 0 for c in ai_share.index}
cluster_to_label[ai_share.idxmax()] = 1
print("\nCluster -> label mapping:", cluster_to_label)

# ---------- Predict on TEST data ----------
test_clusters = kmeans.predict(scaler.transform(test_df[feature_cols]))
test_pred = pd.Series(test_clusters, index=test_df.index).map(cluster_to_label)
y_test = test_df["label"]

# ---------- Evaluate ----------
baseline = (y_test == 0).mean()  # accuracy from always predicting "human"

print("\nAdjusted Rand Index:", adjusted_rand_score(y_test, test_clusters))
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Baseline (always human):", baseline)
print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nClassification report:")
print(classification_report(y_test, test_pred, zero_division=0))