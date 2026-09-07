# Problem Set 01 — Pneumonia Classification Using CNN

## 1. Problem Statement

The objective of this problem is to develop a Convolutional Neural Network
(CNN) model to classify pediatric chest X-ray images into two categories:

- Normal
- Pneumonia

The dataset is organized into training, validation, and testing folders.

## 2. Objective

The goal is to train a CNN that learns visual patterns from chest X-ray images
and predicts whether an X-ray belongs to the Normal or Pneumonia class.

## 3. Dataset

The provided dataset contains chest X-ray images from pediatric patients and
is divided into:

- `train`
- `val`
- `test`

Each split contains two class folders:

- `NORMAL`
- `PNEUMONIA`

The model was trained using the training set, monitored using the validation
set, and evaluated on the unseen test set.

## 4. Approach and Methodology

The workflow used in the project was:

1. Load the chest X-ray image dataset.
2. Inspect the available classes and image distribution.
3. Resize images to a consistent input size.
4. Normalize pixel values.
5. Apply image augmentation to improve generalization.
6. Build a CNN using convolutional and pooling layers.
7. Add dense layers for classification.
8. Use dropout to reduce overfitting.
9. Train the CNN on the training data.
10. Monitor performance using the validation data.
11. Evaluate the trained model on the test data.
12. Generate a confusion matrix and classification metrics.
13. Examine the model's predictions on sample images.

## 5. CNN Model

The model uses convolutional layers to learn spatial features from the
X-ray images. Pooling layers reduce the spatial dimensions of the learned
feature maps.

The extracted features are then passed through fully connected layers.
Dropout is used before the final classification layer.

Because this is a binary classification problem, the final output uses a
sigmoid activation.

## 6. Data Preprocessing

The images were resized to a common image size before being supplied to the
CNN. Pixel values were normalized so that the neural network could train more
effectively.

Image augmentation was also used to expose the model to slightly varied
training examples and reduce the chance of overfitting.

## 7. Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- AUC / ROC-AUC
- Confusion Matrix

For this problem, recall is particularly important to examine because a
pneumonia case incorrectly classified as Normal represents a clinically
important false negative.

## 8. Results

The final test results obtained from the completed notebook were:

| Metric | Result |
|---|---:|
| Test Accuracy | 70.83% |
| Precision | 94.44% |
| Recall | 56.67% |
| F1-score | 70.83% |
| ROC-AUC | 0.8798 |

The test confusion matrix was:

```text
                Predicted
              Normal  Pneumonia

Actual Normal    221       13
Actual Pneumonia 169      221
```

The model therefore produced many correct predictions, but it also missed a
substantial number of pneumonia cases.

## 9. Findings

The model achieved a test accuracy of approximately 70.83% and an ROC-AUC of
0.8798. The high precision indicates that when the model predicted the
positive class, most of those predictions were correct.

However, the recall was 56.67%. The confusion matrix shows that 169 pneumonia
images were classified as Normal. Therefore, accuracy alone does not fully
describe the model's performance.

The results demonstrate that the CNN learned useful information from the
images, but there is still room for improvement, particularly in detecting
pneumonia cases.

## 10. Limitations

The validation split used in this experiment was relatively small compared
with the training and test sets. Therefore, validation performance should be
interpreted carefully.

The model is also an academic machine-learning experiment and should not be
used as a clinical diagnostic system. Real clinical deployment would require
additional validation, appropriate medical oversight, external testing, and
careful assessment of false-negative errors.

## 11. Files

```text
problem_set_01/
├── README.md
├── train_cnn.py
└── requirements.txt
```

## 12. How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then update the dataset paths in `train_cnn.py` and run:

```bash
python train_cnn.py
```

## 13. Conclusion

A CNN-based image classification model was developed to distinguish Normal
and Pneumonia chest X-ray images. The experiment demonstrates the use of
image preprocessing, augmentation, convolutional layers, model training, and
classification metrics for a medical image classification task.
