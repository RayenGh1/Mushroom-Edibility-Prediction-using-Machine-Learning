# Some parts of the code were inspired by examples from:
# Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O’Reilly Media.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import chi2_contingency
from itertools import combinations

# === Load Dataset ===
df = pd.read_csv('Dataset/secondary_data.csv', sep=';').copy()


# === General Dataset Overview ===
print("Dataset Info:")
print(df.info())


print("\nMissing values per column:")
print(df.isnull().sum())

# === Categorical Value Counts ===
for col in df.select_dtypes(include='object').columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts())


# === Descriptive Statistics for Numerical Columns ===
print(df.describe())


# === Encode Categorical Columns Using OrdinalEncoder ===
df_num = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])

cat_pipeline = Pipeline([
    ("encoder", OrdinalEncoder())
])

df_cat_encoded = cat_pipeline.fit_transform(df_cat)
df_cat_encoded = pd.DataFrame(df_cat_encoded, columns=df_cat.columns, index=df.index)


# === Combine Encoded Categorical with Numerical Columns ===
df_encoded = pd.concat([df_num, df_cat_encoded], axis=1)


# === Correlation Matrix (All Features) ===
corr_matrix_full = df_encoded.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix_full, cmap="coolwarm", center=0, cbar=True, annot=True)
plt.title("Correlation Matrix of All Features (Numerical + Encoded Categorical)")
plt.show()


# === Chi² Matrix Between Categorical Features ===
cat_cols = df_cat.columns
chi2_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for col1, col2 in combinations(cat_cols, 2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency)
    chi2_matrix.loc[col1, col2] = chi2
    chi2_matrix.loc[col2, col1] = chi2

np.fill_diagonal(chi2_matrix.values, 0)

plt.figure(figsize=(20, 16))
sns.heatmap(chi2_matrix.astype(float), fmt=".1f", cmap="YlOrRd", square=True)
plt.title("Chi-Square Statistic Matrix Between Categorical Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

pval_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for col1, col2 in combinations(cat_cols, 2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency)
    pval_matrix.loc[col1, col2] = p
    pval_matrix.loc[col2, col1] = p

np.fill_diagonal(pval_matrix.values, 1)

plt.figure(figsize=(20, 16))
sns.heatmap(pval_matrix.astype(float), cmap="YlGnBu", square=True, fmt=".2e")
plt.title("Chi-Square p-value Matrix Between Categorical Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# === Histograms of All Features ===
num_cols = df_encoded.columns
n_cols = 4
n_rows = math.ceil(len(num_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df_encoded[col], bins=30, kde=False, ax=axes[i])
    axes[i].set_title(f"Histogram of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
