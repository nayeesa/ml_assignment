# Pneumonia classification using convolutional neural network (CNN)

## Problem Statement

The objective of this problem is to develop a Convolutional Neural Network  model to classify pediatric chest X-ray images into two categories:
1. Normal
2. Pneumonia

The dataset is organized into training, validation, and testing folders.

## Objective

The goal is to train a CNN that learns visual patterns from chest X-ray images and predicts whether an X-ray belongs to the Normal or Pneumonia class.

## Dataset

The provided dataset contains chest X-ray images from pediatric patients and is divided into:
1. train
2. val
3. test

Each set contains two class folders:
1. NORMAL
2. PNEUMONIA
   
The model was trained using the training set, monitored using the validation set, and evaluated on the unseen test set.

## Methodology

The workflow used in the project was to first load the chest X-ray image dataset, then inspect the available classes and image distribution and resizing images to a consistent input size to normalise pixel values after that we have to apply image augmentation to improve generalisation. Calculate class weights to address class imbalance and build a CNN using convolutional and pooling layers, then add dense layers for classification and use dropout to reduce overfitting. Train the CNN on the training data while monitor performance using the validation data. Evaluate the trained model on the test data while generating a classification report and confusion matrix and calculate the ROC-AUC score. Examine the model's predictions on sample images and finally save the trained model and evaluation results.

## CNN Model

The model uses convolutional layers to learn spatial features from the X-ray images. Pooling layers reduce the spatial dimensions of the learned feature maps. The extracted features are then passed through a Global Average Pooling layer and a fully connected dense layer. Dropout is used before the final classification layer to reduce the risk of overfitting. Because this is a binary classification problem, the final output layer contains one neuron with a sigmoid activation function.

## Data Preprocessing

The images were resized to `224 × 224` pixels before being supplied to the CNN. Pixel values were normalized to the range required for neural network training. Image augmentation was also applied to the training data.  Class weights were also calculated from the training data to reduce the effect of class imbalance during training.

## Evaluation

The model was evaluated using:
1. Accuracy
2. Precision
3. Recall
4. F1-score
5. ROC-AUC
6. Confusion Matrix

## Results

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
  
## Limitations

The validation split used in this experiment was relatively small compared with the training and test sets. Therefore, validation performance should be interpreted carefully.  The model also produced a relatively large number of false negatives, with 169 pneumonia images incorrectly classified as Normal. This is an important limitation because failing to identify pneumonia is more concerning.
The model should not be used as a clinical diagnostic system. Because real clinical system need more validation, medical oversight, external testing, and careful assessment of false-negative errors.

## Conclusion

The CNN model successfully learned visual patterns from pediatric chest X-ray images and was able to distinguish between Normal and Pneumonia cases to a reasonable extent. The model achieved a test accuracy of 70.83% and a ROC-AUC of 0.8798, indicating that it learned meaningful patterns from the dataset. The high precision of 94.44% shows that predictions of Pneumonia were generally reliable.
But the pneumonia recall of 56.67% shows that the model missed a considerable number of pneumonia cases.
Overall, this project demonstrates the use of a CNN-based image classification approach for pediatric chest X-ray images. Future improvements could include using transfer learning with pretrained models, increasing and balancing the training data, applying more advanced preprocessing and augmentation techniques, tuning the CNN architecture and hyperparameters, and optimizing the classification threshold to improve pneumonia recall.

## Files

```text
problem_set_01/
├── README.md
├── train_cnn.py
└── requirements.txt
```

## How to Run

1. Install the required Python packages
2. Update the dataset paths in `train_cnn.py` according to the location of the dataset and run it.

It trains the CNN, evaluates it on the test dataset, generates evaluation metrics and visualizations, and saves the trained model and results.

