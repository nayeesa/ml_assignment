```python
"""
Bank Marketing Prediction using Logistic Regression

Dataset file need:
    bank-full.csv

"""
# import libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# load dataset

df = pd.read_csv("bank-full.csv", sep=";")

print("Dataset loaded.")
print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

# data explore

print("\nDataset information:")
df.info()

print("\nstatistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget variable counts:")
print(df["y"].value_counts())

print("\nTarget variable percent:")
print(df["y"].value_counts(normalize=True) * 100)

# target variable

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="y")

plt.title("Term Deposit Subscription")
plt.xlabel("Subscribed")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()

# separate target and features

X = df.drop("y", axis=1)

y = df["y"].map({
    "no": 0,
    "yes": 1
})

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# identify feature type

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# Train-Test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain data:", X_train.shape)
print("Test data:", X_test.shape)

# preprocessing 

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# pipeline 

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

# train model

print("\nTrain Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Model train completed.")

# prediction and evaluate metrics

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

# result 

print("\n===== FINAL MODEL RESULTS =====")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# report 

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No", "Yes"]
    )
)

# Confusion matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n== CONFUSION MATRIX ==")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")

plt.tight_layout()
plt.show()

#  ROC curve

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")

plt.legend()
plt.tight_layout()
plt.show()

# save Results

results = {
    "Model": "Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC-AUC": roc_auc
}

results_df = pd.DataFrame([results])

results_df.to_csv(
    "logistic_regression_results.csv",
    index=False
)

print("\nResults saved to:")
print("logistic_regression_results.csv")

print("\n==RESULTS TABLE ==")
print(results_df)

print("\nModel execution completed.")
