import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)

# Settings
TRAIN_PATH = "data/features/drcat_train_final_sentence_features.csv"
CLASS_LABEL = 0        # the single class to cluster: 1 = AI, 0 = human
N_CLUSTERS = 4         # no sharp elbow in this data, so check the elbow plot below
N_EXAMPLES = 4         # example sentences printed per cluster

feature_cols = [
    "n_words", "long_word_ratio", "func_word_ratio", "unique_word_ratio",
    "comma_rate", "exclaim", "apostrophe_rate", "starts_with_transition",
    "rel_len_deviation", "noun_ratio", "adjective_ratio", "adverb_ratio",
    "pronoun_ratio", "auxiliary_ratio", "conjunction_ratio",
]

# Metadata columns, not used for clustering
check_cols = ["generator", "domain", "source_dataset", "label_origin"]

# 1. Select one class
full_df = pd.read_csv(TRAIN_PATH)
df = full_df[full_df["label"] == CLASS_LABEL].reset_index(drop=True)
print(f"Clustering {len(df):,} sentences with label {CLASS_LABEL}")

# 2. Scale and cluster
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

joblib.dump(scaler, "scaler_single_class.joblib")
joblib.dump(kmeans, "kmeans_single_class.joblib")

# Distance of every sentence to its own cluster centre
dists = np.linalg.norm(X_scaled - kmeans.cluster_centers_[df["cluster"]], axis=1)
df["dist_to_center"] = dists

# 3. Cluster sizes
print("\n=== Cluster sizes ===")
sizes = df["cluster"].value_counts().sort_index()
print(pd.DataFrame({"n": sizes, "share": (sizes / len(df)).round(3)}))

# 4. Compare cluster mean to overall mean
overall_mean = df[feature_cols].mean()
overall_std = df[feature_cols].std()
overall_zero = (df[feature_cols] == 0).mean()

cluster_mean = df.groupby("cluster")[feature_cols].mean()
z_diff = (cluster_mean - overall_mean) / overall_std

print("\n=== Cluster descriptions ===")
for c in sorted(df["cluster"].unique()):
    members = df[df["cluster"] == c]
    print(f"\n----- Cluster {c}  (n={len(members):,}, {len(members) / len(df):.1%}) -----")

    # Most distinctive features
    top = z_diff.loc[c].abs().sort_values(ascending=False).head(3).index
    print("Most distinctive features (cluster mean vs overall mean):")
    for f in top:
        direction = "higher" if z_diff.loc[c, f] > 0 else "lower"
        print(f"  {f:24s} {cluster_mean.loc[c, f]:.3f} vs {overall_mean[f]:.3f}  "
              f"({direction}, {z_diff.loc[c, f]:+.2f} SD)")

    # Missing features
    zero_share = (members[feature_cols] == 0).mean()
    missing = (zero_share - overall_zero).sort_values(ascending=False)
    missing = missing[missing > 0.10].head(3)
    if len(missing):
        print("What is missing (share of sentences where the feature is zero):")
        for f, diff in missing.items():
            print(f"  {f:24s} zero in {zero_share[f]:.0%} of this cluster "
                  f"vs {overall_zero[f]:.0%} overall")

    # Examples
    print("Representative sentences:")
    for text in members.nsmallest(N_EXAMPLES, "dist_to_center")["text"]:
        text = str(text).replace("\\n", " ").strip()
        print(f"  - {text[:200]}")

# 5. Save summary table
summary = cluster_mean.copy()
summary.insert(0, "n", sizes)
summary.to_csv("cluster_summary.csv")
print("\nSaved cluster_summary.csv")

# 6. Plot
top2 = z_diff.std().sort_values(ascending=False).head(2).index.tolist()

plt.figure(figsize=(8, 6))
plt.scatter(df[top2[0]], df[top2[1]], c=df["cluster"], s=1, alpha=0.2, cmap="tab10")
plt.xlabel(top2[0])
plt.ylabel(top2[1])
plt.title(f"K-Means clusters within class {CLASS_LABEL}")
plt.show()