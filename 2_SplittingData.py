# Some parts of the code were inspired by examples from:
# Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O’Reilly Media.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# === Load Dataset ===
df = pd.read_csv('Dataset/secondary_data.csv', sep=';')
df = df.drop(labels="veil-type", axis=1)


# === Create Binned Version of 'stem-width' for Stratification ===
stem_width = df["stem-width"].copy()
bins = list(range(0, 51)) + [np.inf]
labels = [str(i) for i in range(0, 50)] + [">50"]

df["stem-width-binned"] = pd.cut(stem_width, bins=bins, labels=labels, right=False)


# === Visualize Distribution of Binned 'stem-width' ===
print(df["stem-width-binned"].value_counts().sort_index())
df["stem-width-binned"].value_counts().sort_index().plot(kind="bar", figsize=(10, 6))
plt.title("Stem-width (binned: 0–49 + >50)")
plt.xlabel("Stem-width category")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# === Stratified Split Based on Binned 'stem-width' ===
strat_train_set, strat_test_set = train_test_split(
df, test_size=0.2, stratify=df["stem-width-binned"], random_state=42
)


# === Drop Temporary Binned Column After Splitting ===
for dataset in (strat_train_set, strat_test_set):
    dataset.drop("stem-width-binned", axis=1, inplace=True)


# === Save the Stratified Train and Test Sets ===
strat_train_set.to_csv("Dataset/strat_train_set.csv", index=False)
strat_test_set.to_csv("Dataset/strat_test_set.csv", index=False)
