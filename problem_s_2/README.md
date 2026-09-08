# Problem Set 02 — Bank Marketing Prediction Using Logistic Regression

## 1. Problem Statement

The objective of this problem is to build a machine learning model that predicts whether a customer will subscribe to a term deposit based on information collected during a bank marketing campaign.

The target variable is `y`, which contains two possible outcomes:

* `yes` — the customer subscribed to a term deposit
* `no` — the customer did not subscribe

A Logistic Regression model is used to perform this binary classification task.

---

## 2. Objective

The main objectives are to:

1. Load and explore the Bank Marketing dataset.
2. Perform basic data inspection and preprocessing.
3. Separate the input features from the target variable.
4. Encode categorical variables.
5. Scale numerical variables.
6. Train a Logistic Regression classification model.
7. Evaluate the model using multiple classification metrics.
8. Analyze the model's strengths and limitations.

---

## 3. Dataset

The dataset used is the **Bank Marketing dataset**.

It contains information about customers contacted during a bank marketing campaign.

The dataset contains:

* **45,211 records**
* **17 columns**

The target variable is:

* `y` — whether the customer subscribed to a term deposit.

The input variables include customer and campaign-related attributes such as:

* `age`
* `job`
* `marital`
* `education`
* `default`
* `balance`
* `housing`
* `loan`
* `contact`
* `day`
* `month`
* `duration`
* `campaign`
* `pdays`
* `previous`
* `poutcome`

---

## 4. Approach and Methodology

### Step 1: Data Loading

The dataset was loaded using Pandas. Since the original dataset uses semicolons as separators, `sep=";"` was used when reading the CSV file.

```python
df = pd.read_csv("bank-full.csv", sep=";")
```

### Step 2: Data Exploration

The dataset was inspected using:

* Dataset shape
* Data types
* Descriptive statistics
* Missing-value checks
* Target-variable distribution

The target distribution was also visualized using a count plot.

### Step 3: Feature and Target Separation

The target column `y` was separated from the input features.

The target values were converted into binary numerical values:

* `no` → `0`
* `yes` → `1`

### Step 4: Identifying Feature Types

The input features were divided into:

* Numerical features
* Categorical features

This allowed different preprocessing methods to be applied to each type of variable.

### Step 5: Data Splitting

The dataset was divided into training and testing sets using an 80/20 split.

Stratified sampling was used so that the class distribution was maintained between the training and testing datasets.

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Step 6: Data Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were converted into numerical representations using `OneHotEncoder`.

Unknown categories were ignored during encoding to make the preprocessing pipeline more robust.

### Step 7: Model Training

A Logistic Regression classifier was used.

The preprocessing and classification steps were combined into a single Scikit-learn Pipeline.

The Logistic Regression model was configured with:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The pipeline was then fitted using the training data.

---

## 5. Model

The final machine learning pipeline consists of:

1. `StandardScaler` for numerical features
2. `OneHotEncoder` for categorical features
3. `LogisticRegression` for binary classification

The pipeline ensures that preprocessing is consistently applied before making predictions.

---

## 6. Evaluation Metrics

The model was evaluated using the following metrics:

### Accuracy

Measures the overall proportion of correctly classified customers.

### Precision

Measures how many customers predicted as subscribers actually subscribed.

### Recall

Measures how many of the actual subscribers were correctly identified.

### F1 Score

Provides a balance between precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between customers who subscribed and those who did not across different classification thresholds.

A confusion matrix and ROC curve were also generated for further evaluation.

---

## 7. Results

The Logistic Regression model produced the following results on the test dataset:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.9012 |
| Precision | 0.6445 |
| Recall    | 0.3478 |
| F1 Score  | 0.4518 |
| ROC-AUC   | 0.9056 |

### Interpretation

The model achieved an accuracy of **90.12%**, which indicates that most test observations were classified correctly.

The **precision of 64.45%** means that when the model predicts that a customer will subscribe, the prediction is correct in a substantial proportion of cases.

However, the **recall of 34.78%** is considerably lower. This means that the model identifies only a portion of the customers who actually subscribe to the term deposit.

The **F1 score of 45.18%** reflects the imbalance between precision and recall.

The **ROC-AUC score of 0.9056** is strong and indicates that the model has good ability to distinguish between the two target classes.

---

## 8. Findings

Several observations can be made from the results:

1. The model achieved high overall accuracy at **90.12%**.
2. The ROC-AUC score of **0.9056** indicates strong class-separation ability.
3. Precision was **64.45%**, meaning positive predictions were reasonably reliable.
4. Recall was only **34.78%**, showing that many actual subscribers were not identified by the default classification threshold.
5. The difference between accuracy and recall suggests that accuracy alone is not sufficient for evaluating this problem.
6. For a marketing campaign, improving recall could be important because missing potential subscribers may reduce the effectiveness of customer targeting.

---

## 9. Limitations

The model has several limitations.

First, the dataset contains a substantially larger number of customers who did not subscribe than customers who did. Therefore, accuracy can give an overly optimistic impression of performance.

Second, the recall of **34.78%** indicates that many positive cases are missed by the model.

Third, the model uses the default classification threshold for converting predicted probabilities into class predictions. Changing this threshold could potentially improve recall depending on the business objective.

Finally, Logistic Regression is a relatively simple classification algorithm. More complex models could potentially capture nonlinear relationships between customer characteristics and subscription behavior.

---

## 10. Conclusion

A Logistic Regression model was developed to predict whether bank customers would subscribe to a term deposit.

The model achieved **90.12% accuracy** and a **ROC-AUC of 0.9056**, demonstrating strong overall classification and class-separation performance.

However, the relatively low **34.78% recall** shows that the model fails to identify a significant number of customers who actually subscribe. Therefore, while the model performs well in overall discrimination, further work would be useful to improve the identification of positive customers.

Possible future improvements include experimenting with classification thresholds, handling class imbalance, feature selection, and comparing Logistic Regression with other classification algorithms.

---

## 11. Files

```text
problem_set_02/
├── README.md
├── train_logistic_regression.py
└── requirements.txt
```

The original dataset is not included in the Git repository because it is provided separately and does not need to be uploaded with the solution.

---

## 12. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the program

```bash
python train_logistic_regression.py
```

The program loads the Bank Marketing dataset, performs preprocessing, trains the Logistic Regression model, and evaluates its performance.

The dataset file should be placed in the appropriate location expected by the Python script.

