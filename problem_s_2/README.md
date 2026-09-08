# Bank Marketing Prediction using Logistic Regression

## Objective

first have to load and explore the Bank Marketing dataset. Performing basic data inspection and preprocessing while separate the input features from the target variable. Encode the categorical variables and scale numerical variables. Train a Logistic Regression classification model and evaluate the model using multiple classification metrics.finally Analyze the model's strengths and limitations.

## Dataset

The dataset used is the **Bank Marketing dataset**. It contains information about customers contacted during a bank marketing campaign.
The dataset contains:
1. 45,211 records
2. 17 columns

The target variable is `y` to see if the customer subscribed to a term deposit.

The input variables include customer and campaign-related attributes such as:
`age`,`job`, `marital`, `education`, `default`, `balance`, `housing`,`loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous` and `poutcome`

## Methodology

The dataset was loaded using Pandas. Since the original dataset uses semicolons as separators, `sep=";"` was used when reading the CSV file.
The dataset was inspected using Dataset shape, Data types, Descriptive statistics, Missing-value checks, Target-variable distribution.
The target column `y` was separated from the input features. The target values were converted into binary numerical values:
1. `no` → `0`
2. `yes` → `1`

The input features were divided into numerical features and categorical features. The dataset was divided into training and testing sets using an 80/20 split. Numerical features were standardized using `StandardScaler`and categorical features were converted into numerical representations using `OneHotEncoder`. Logistic Regression classifier was used to train the model. The pipeline was then fitted using the training data.

## Evaluation Metrics

The model was evaluated using Accuracy, Precision, Recall, F1 Score, ROC-AUC.A confusion matrix and ROC curve were also generated for further evaluation.

## Results

The Logistic Regression model produced the following results on the test dataset:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.9012 |
| Precision | 0.6445 |
| Recall    | 0.3478 |
| F1 Score  | 0.4518 |
| ROC-AUC   | 0.9056 |

## Limitations

The model has several limitations.
First, the dataset contains a substantially larger number of customers who did not subscribe than customers who did. 
Second, the recall of **34.78%** indicates that many positive cases are missed.
Third, the model uses the default classification threshold for converting predicted probabilities into class predictions.
Finally, More complex models could potentially capture nonlinear relationships between customer characteristics and subscription behavior.

## Conclusion

A Logistic Regression model was developed to predict whether bank customers would subscribe to a term deposit. The model achieved **90.12% accuracy** and a **ROC-AUC of 0.9056**.
But the relatively low **34.78% recall** shows that the model fails to identify a significant number of customers who actually subscribe. Possible future improvements include experimenting with classification thresholds, handling class imbalance, feature selection, and comparing Logistic Regression with other classification algorithms.

## Files

```text
problem_set_02/
├── README.md
├── train_logistic_regression.py
└── requirements.txt
```

## How to Run

1. Install dependencies
2. Run the program

It loads the Bank Marketing dataset, performs preprocessing, trains the Logistic Regression model, and evaluates its performance.
