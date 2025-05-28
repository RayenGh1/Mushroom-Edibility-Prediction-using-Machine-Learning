# Some parts of the code were inspired by examples from:
# Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O’Reilly Media.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# === Load data ===
train_df = pd.read_csv('Dataset/strat_train_set.csv')
test_df = pd.read_csv('Dataset/strat_test_set.csv')

x_train = train_df.drop("class", axis=1)
y_train = train_df["class"].copy()
x_test = test_df.drop("class", axis=1)
y_test = test_df["class"]

skf = StratifiedKFold(n_splits= 5)


# === Check for data leakage ===
print("\n=== Column names in training set ===")
print(train_df.columns)


# Calculate correlation between numeric features and the encoded class
corr = train_df.copy()
corr["class_encoded"] = corr["class"].astype("category").cat.codes
correlation_with_target = corr.corr(numeric_only=True)["class_encoded"].sort_values(ascending=False)

print("\n=== Correlation of numerical column with 'class' ===")
print(correlation_with_target)

for col in train_df.select_dtypes(exclude=[np.number]).columns:
    if col != "class":
        crosstab = pd.crosstab(train_df[col], train_df["class"], normalize="index")
        print(f"\ncolumn: {col}")
        print(crosstab.sort_values(by=crosstab.columns[0], ascending=False).head())


# === Preprocessing ===
num_cols = x_train.select_dtypes(include=[np.number]).columns
cat_cols = x_train.select_dtypes(exclude=[np.number]).columns

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # Replace missing values with the median
    ("scaler", StandardScaler()) # Scale to unit variance
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


# === Define models and fine-tuning ===
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNeighbors": KNeighborsClassifier(),
    "Support Vector": SVC(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SGD": SGDClassifier(random_state=42)
}

param_grids = {
    "Decision Tree": {
        "decisiontreeclassifier__max_depth": [3, 5, 10, 20, None],
        "decisiontreeclassifier__min_samples_split": [2, 5, 10, 20]
    },
    "Random Forest": {
        "randomforestclassifier__n_estimators": [50, 100, 200],
        "randomforestclassifier__max_depth": [3, 5, 10, 20, None]
    },
    "KNeighbors": {
        "kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9],
        "kneighborsclassifier__weights": ['uniform', 'distance']
    }
}

trained_models = {}



# === Train each model ===
for name, model in models.items():
    
    print(f"\n--- Training model: {name} ---")

    # === Base model ===
    model_class = make_pipeline(preprocessing, model)
    start_time = time.time()
    model_class.fit(x_train, y_train)
    end_time = time.time()
    duration = end_time - start_time

    y_pred_test = model_class.predict(x_test)
    acc = accuracy_score(y_test, y_pred_test)
    cv_scores = cross_val_score(model_class, x_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_mean = cv_scores.mean()

    # Save base model performance
    trained_models[name + " (base)"] = {
        "model": model_class,
        "y_pred_test": y_pred_test,
        "test_accuracy": acc,
        "cv_accuracy": cv_mean,
        "training_time": duration,
        "preprocessing": "norm + onehot",
        "data_split": "random",
        "fine_tuned": False,
        "best_params": None
    }

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_class.classes_)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {name} (base)")
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/conf_matrix_{name.replace(' ', '_').lower()}_base.png")
    plt.close()

    print("\n=== Detailed classification report ===")
    print(f" {name} (base)")
    print(classification_report(y_test, y_pred_test, digits=4))

    # === Print important variables per model ===
    print("\n=== Important variables per model ===")
    feature_names = model_class.named_steps["columntransformer"].get_feature_names_out()
    classifier = model_class.named_steps[list(model_class.named_steps.keys())[-1]]
    importance = None

    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_).mean(axis=0)
        
    if importance is not None:
        # Print top 10
        importance_series = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        print(f"\n {name} - Top 10 important features:")
        print(importance_series.head(10))
    else:
        print(f"\n {name} support no feature importance")

    # === Fine-tuning using GridSearchCV ===
    if name in param_grids:
        print("   Fine-tuning with GridSearchCV...")
        grid_search = GridSearchCV(model_class, param_grids[name], cv=5, scoring='accuracy')
        start_time = time.time()
        grid_search.fit(x_train, y_train)
        end_time = time.time()
        final_model = grid_search.best_estimator_
        duration = end_time - start_time

        y_pred_test = final_model.predict(x_test)
        acc = accuracy_score(y_test, y_pred_test)
        cv_scores = cross_val_score(final_model, x_train, y_train, cv=5, scoring="accuracy")
        cv_mean = cv_scores.mean()

        trained_models[name + " (tuned)"] = {
            "model": final_model,
            "y_pred_test": y_pred_test,
            "test_accuracy": acc,
            "cv_accuracy": cv_mean,
            "training_time": duration,
            "preprocessing": "norm + onehot",
            "data_split": "random",
            "fine_tuned": True,
            "best_params": grid_search.best_params_
        }

        # Save confusion matrix for tuned model
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
        disp.plot(xticks_rotation=45)
        plt.title(f"Confusion Matrix - {name} (tuned)")
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/conf_matrix_{name.replace(' ', '_').lower()}_tuned.png")
        plt.close()

        print("\n=== Detailed classification report ===")
        print(f"-> {name} (tuned)")
        print(classification_report(y_test, y_pred_test, digits=4))

# === Show comparison table ===
print("\n=== Comparison Table ===")
comparison_table = pd.DataFrame([
    {
        "method": name,
        "pre-processing step": info["preprocessing"],
        "data split": info["data_split"],
        "accuracy (cv)": f"{info['cv_accuracy']:.4f}",
        "accuracy (test)": f"{info['test_accuracy']:.4f}",
        "training time (s)": f"{info['training_time']:.2f}",
        "best params": info["best_params"]
    }
    for name, info in trained_models.items()
])

pd.set_option('display.max_columns', None) # To print all columns (because screen is too small)
pd.set_option('display.max_colwidth', None)
print(comparison_table)
print(y_test.value_counts(normalize=True))

print("\n=== Predictions on test set per model ===")
for name, info in trained_models.items():
    print(f"\n-> Predictions from: {name}")
    print(pd.Series(info["y_pred_test"]).value_counts())


# === Save all trained models ===
os.makedirs("saved_models", exist_ok=True)

for name, info in trained_models.items():
    filename = f"saved_models/{name.replace(' ', '_').replace('(', '').replace(')', '')}.pkl"
    joblib.dump(info["model"], filename)
    print(f"Saved model: {filename}")