# Problem Set 01 — Pneumonia Classification Using CNN

## 1. Problem Statement

The objective of this problem is to develop a Convolutional Neural Network (CNN) model to classify pediatric chest X-ray images into two categories:

* Normal
* Pneumonia

The dataset is organized into training, validation, and testing folders.

## 2. Objective

The goal is to train a CNN that learns visual patterns from chest X-ray images and predicts whether an X-ray belongs to the Normal or Pneumonia class.

## 3. Dataset

The provided dataset contains chest X-ray images from pediatric patients and is divided into:

* `train`
* `val`
* `test`

Each split contains two class folders:

* `NORMAL`
* `PNEUMONIA`

The model was trained using the training set, monitored using the validation set, and evaluated on the unseen test set.

## 4. Approach and Methodology

The workflow used in the project was:

1. Load the chest X-ray image dataset.
2. Inspect the available classes and image distribution.
3. Resize images to a consistent input size.
4. Normalize pixel values.
5. Apply image augmentation to improve generalization.
6. Calculate class weights to address class imbalance.
7. Build a CNN using convolutional and pooling layers.
8. Add dense layers for classification.
9. Use dropout to reduce overfitting.
10. Train the CNN on the training data.
11. Monitor performance using the validation data.
12. Evaluate the trained model on the test data.
13. Generate a classification report and confusion matrix.
14. Calculate the ROC-AUC score.
15. Examine the model's predictions on sample images.
16. Save the trained model and evaluation results.

## 5. CNN Model

The model uses convolutional layers to learn spatial features from the X-ray images. Pooling layers reduce the spatial dimensions of the learned feature maps.

The CNN consists of four convolutional blocks with increasing numbers of filters. The extracted features are then passed through a Global Average Pooling layer and a fully connected dense layer.

Dropout is used before the final classification layer to reduce the risk of overfitting.

Because this is a binary classification problem, the final output layer contains one neuron with a sigmoid activation function.

The model was optimized using the Adam optimizer with a learning rate of `0.0001` and binary cross-entropy as the loss function.

## 6. Data Preprocessing

The images were resized to `224 × 224` pixels before being supplied to the CNN. Pixel values were normalized to the range required for neural network training.

Image augmentation was also applied to the training data. The augmentation included:

* Random horizontal flipping
* Random rotation
* Random zoom

These transformations were used to expose the model to slightly varied training examples and help improve generalization.

Class weights were also calculated from the training data to reduce the effect of class imbalance during training.

## 7. Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

For this problem, recall is particularly important because a pneumonia case incorrectly classified as Normal represents a clinically important false negative.

ROC-AUC was also used to measure the model's ability to distinguish between the two classes across different classification thresholds.

## 8. Results

The final test results obtained from the completed notebook were:

| Metric        | Result |
| ------------- | -----: |
| Test Accuracy | 70.83% |
| Precision     | 94.44% |
| Recall        | 56.67% |
| F1-score      | 70.83% |
| ROC-AUC       | 0.8798 |

The test confusion matrix was:

```text
                 Predicted
               Normal  Pneumonia

Actual Normal     221       13
Actual Pneumonia  169      221
```

The model correctly classified:

* 221 Normal images as Normal.
* 221 Pneumonia images as Pneumonia.

It incorrectly classified:

* 13 Normal images as Pneumonia.
* 169 Pneumonia images as Normal.

Therefore, the model produced many correct predictions, but it also missed a substantial number of pneumonia cases.

## 9. Findings

The model achieved a test accuracy of approximately 70.83% and an ROC-AUC of 0.8798. The high precision of 94.44% indicates that when the model predicted Pneumonia, most of those predictions were correct.

However, the recall was 56.67%. The confusion matrix shows that 169 out of 390 pneumonia images were classified as Normal. This means that the model had difficulty identifying a significant portion of the pneumonia cases.

Therefore, accuracy alone does not fully describe the model's performance. The ROC-AUC score of 0.8798 suggests that the model learned useful patterns for distinguishing between Normal and Pneumonia images, even though its default classification threshold resulted in a relatively low pneumonia recall.

The results demonstrate that the CNN successfully learned meaningful visual information from the X-ray images, but there is still considerable room for improvement, particularly in detecting pneumonia cases.

## 10. Limitations

The validation split used in this experiment was relatively small compared with the training and test sets. Therefore, validation performance should be interpreted carefully.

The model also produced a relatively large number of false negatives, with 169 pneumonia images incorrectly classified as Normal. This is an important limitation because failing to identify pneumonia is more concerning than incorrectly flagging a Normal image as Pneumonia.

The model is an academic machine-learning experiment and should not be used as a clinical diagnostic system. Real clinical deployment would require additional validation, appropriate medical oversight, external testing, and careful assessment of false-negative errors.

## 11. Conclusion

The CNN model successfully learned visual patterns from pediatric chest X-ray images and was able to distinguish between Normal and Pneumonia cases to a reasonable extent.

The model achieved a test accuracy of 70.83% and a ROC-AUC of 0.8798, indicating that it learned meaningful patterns from the dataset. The high precision of 94.44% shows that predictions of Pneumonia were generally reliable.

However, the pneumonia recall of 56.67% shows that the model missed a considerable number of pneumonia cases. This demonstrates why multiple evaluation metrics are important for assessing a classification model, especially for applications where false-negative predictions can be significant.

Overall, this project demonstrates the use of a CNN-based image classification approach for pediatric chest X-ray images. Future improvements could include using transfer learning with pretrained models, increasing and balancing the training data, applying more advanced preprocessing and augmentation techniques, tuning the CNN architecture and hyperparameters, and optimizing the classification threshold to improve pneumonia recall.

## 12. Files

```text
problem_set_01/
├── README.md
├── train_cnn.py
└── requirements.txt
```

## 13. How to Run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then update the dataset paths in `train_cnn.py` according to the location of the dataset and run:

```bash
python train_cnn.py
```

The script trains the CNN, evaluates it on the test dataset, generates evaluation metrics and visualizations, and saves the trained model and results.

