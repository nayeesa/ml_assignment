"""
Pneumonia Chest X-Ray Classification using a Convolutional Neural Network (CNN)

This script is a cleaned, GitHub-friendly version of the code developed in
the accompanying Colab notebook. Update the dataset paths below to match
your local/Colab environment before running.
"""




# ======================================================================
# Notebook cell
# ======================================================================import os

folder = "/content/drive/MyDrive/pneumonia_cnn"

print("Files in your folder:")
for file in os.listdir(folder):
    print(file)

import os

folder = "/content/drive/MyDrive/pneumonia_cnn"

print("Files in your folder:")
print(os.listdir(folder))

import os

zip_path = "/content/drive/MyDrive/pneumonia_cnn/Archive.zip"

size_gb = os.path.getsize(zip_path) / (1024 ** 3)

print(f"ZIP file size: {size_gb:.2f} GB")

import zipfile
import os

zip_path = "/content/drive/MyDrive/pneumonia_cnn/Archive.zip"
extract_path = "/content/chest_xray"

print("Starting extraction...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction completed!")

import os

print("Contents of /content/chest_xray:")

for item in os.listdir("/content/chest_xray"):
    print(item)

from pathlib import Path

base = Path("/content/chest_xray")

print("Searching for dataset folders...\n")

for folder_name in ["train", "test", "val"]:
    matches = list(base.rglob(folder_name))

    print(f"{folder_name}:")
    for match in matches:
        print("  ", match)

from pathlib import Path

base = Path("/content/chest_xray")

train_matches = list(base.rglob("train"))
test_matches = list(base.rglob("test"))
val_matches = list(base.rglob("val"))

if not train_matches:
    raise FileNotFoundError("Could not find the train folder.")

if not test_matches:
    raise FileNotFoundError("Could not find the test folder.")

if not val_matches:
    raise FileNotFoundError("Could not find the val folder.")

train_dir = train_matches[0]
test_dir = test_matches[0]
val_dir = val_matches[0]

print("Train:", train_dir)
print("Test :", test_dir)
print("Val  :", val_dir)

import os

print("Training classes:")
print(os.listdir(train_dir))

print("\nTesting classes:")
print(os.listdir(test_dir))

print("\nValidation classes:")
print(os.listdir(val_dir))

from pathlib import Path

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png"}

    counts = {}

    for class_folder in folder.iterdir():
        if class_folder.is_dir():
            count = sum(
                1 for file in class_folder.rglob("*")
                if file.suffix.lower() in extensions
            )
            counts[class_folder.name] = count

    return counts

print("TRAIN:")
print(count_images(train_dir))

print("\nVALIDATION:")
print(count_images(val_dir))

print("\nTEST:")
print(count_images(test_dir))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("Image size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

print("Training dataset loaded.")
print("Classes:", train_ds.class_names)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Validation dataset loaded.")
print("Classes:", val_ds.class_names)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Test dataset loaded.")
print("Classes:", test_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Dataset pipeline optimized.")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
], name="data_augmentation")

print("Data augmentation created.")

# Calculate class weights

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Get class names from the dataset before prefetching
class_names = sorted([
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
])

print("Classes found:", class_names)

# Count images in each class
class_counts = {}

for class_name in class_names:
    class_folder = train_dir / class_name

    count = sum(
        1
        for file in class_folder.rglob("*")
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    class_counts[class_name] = count

print("\nTraining image counts:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Create labels
labels = []

for class_index, class_name in enumerate(class_names):
    labels.extend([class_index] * class_counts[class_name])

labels = np.array(labels)

# Calculate balanced class weights
classes = np.unique(labels)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels
)

class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(classes, weights)
}

print("\nClass weights:")
print(class_weights)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("Model compiled successfully.")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "/content/best_pneumonia_cnn.keras",
    monitor="val_loss",
    save_best_only=True
)

print("Callbacks created.")

EPOCHS = 15

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.show()

test_results = model.evaluate(test_ds, verbose=1)

print("\nTest results:")

for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

y_true = []
y_prob = []

for images, labels in test_ds:
    probabilities = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten())
    y_prob.extend(probabilities.flatten())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = (y_prob >= 0.5).astype(int)

print("Predictions generated.")
print("Number of test images:", len(y_true))

# Classification report

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# Display confusion matrix

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()
plt.title("Confusion Matrix")
plt.show()

auc_score = roc_auc_score(y_true, y_prob)

print(f"ROC-AUC: {auc_score:.4f}")

model_path = "/content/drive/MyDrive/pneumonia_cnn/pneumonia_cnn_model.keras"

model.save(model_path)

print("Model saved successfully!")
print(model_path)

import json

results = {
    "test_accuracy": float(test_results[1]),
    "test_auc": float(test_results[2]),
    "test_precision": float(test_results[3]),
    "test_recall": float(test_results[4]),
    "roc_auc": float(auc_score)
}

results_path = "/content/drive/MyDrive/pneumonia_cnn/results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print("Results saved to:")
print(results_path)

print("\nResults:")
print(json.dumps(results, indent=4))

# Show sample predictions

import matplotlib.pyplot as plt

images, labels = next(iter(test_ds))

probabilities = model.predict(images, verbose=0).flatten()

plt.figure(figsize=(12, 10))

for i in range(min(9, len(images))):

    plt.subplot(3, 3, i + 1)

    plt.imshow(images[i].numpy().astype("uint8"))

    actual = class_names[int(labels[i].numpy())]

    predicted_index = int(probabilities[i] >= 0.5)

    predicted = class_names[predicted_index]

    confidence = (
        probabilities[i]
        if predicted_index == 1
        else 1 - probabilities[i]
    )

    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence:.2f}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()