import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features/drcat_train_final_sentence_features.csv")

feature_cols = [
    "n_words", "long_word_ratio", "func_word_ratio", "unique_word_ratio",
    "comma_rate", "exclaim", "apostrophe_rate", "starts_with_transition",
    "rel_len_deviation", "noun_ratio", "adjective_ratio", "adverb_ratio",
    "pronoun_ratio", "auxiliary_ratio", "conjunction_ratio",
]

X = df[feature_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_scaled)

joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans.joblib")

plt.figure(figsize=(8, 6))
plt.scatter(df["long_word_ratio"], df["func_word_ratio"],
            c=df["cluster"], s=1, alpha=0.2)
plt.xlabel("long_word_ratio")
plt.ylabel("func_word_ratio")
plt.title("K-Means clusters")
plt.show()