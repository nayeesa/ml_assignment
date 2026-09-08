"""
Pneumonia Chest X-Ray Classification using a Convolutional Neural Network (CNN)

This script trains a CNN to classify pediatric chest X-ray images as
NORMAL or PNEUMONIA.
"""

# IMPORT LIBRARIES

import os
import json
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from google.colab import drive

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

# Mount google drive 

drive.mount("/content/drive")

# DATASET paths and extracting

drive_folder = "/content/drive/MyDrive/pneumonia_cnn"

zip_path = os.path.join(drive_folder, "Archive.zip")

# Dataset will be extracted here for faster training
extract_path = Path("/content/chest_xray")

print("Dataset ZIP:", zip_path)

if not os.path.exists(zip_path):
    raise FileNotFoundError(
        f"Dataset ZIP file was not found at:\n{zip_path}"
    )

if not extract_path.exists():
    print("Extracting dataset...")
    
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    
    print("Dataset extraction completed.")
else:
    print("Dataset already extracted. Skipping extraction.")


# ============================================================
# 5. FIND TRAIN, VALIDATION AND TEST FOLDERS
# ============================================================

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

print("\nDataset directories:")
print("Train:", train_dir)
print("Validation:", val_dir)
print("Test:", test_dir)


# ============================================================
# 6. COUNT IMAGES
# ============================================================

def count_images(folder):
    """Count image files in each class folder."""
    
    extensions = {".jpg", ".jpeg", ".png"}
    counts = {}

    for class_folder in folder.iterdir():
        if class_folder.is_dir():
            count = sum(
                1
                for file in class_folder.rglob("*")
                if file.suffix.lower() in extensions
            )

            counts[class_folder.name] = count

    return counts


print("\nTraining images:")
print(count_images(train_dir))

print("\nValidation images:")
print(count_images(val_dir))

print("\nTesting images:")
print(count_images(test_dir))


# ============================================================
# 7. CONFIGURATION
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

print("\nImage size:", IMG_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nTensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 8. LOAD DATASETS
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

# Class names are saved before modifying the dataset pipeline.
class_names = train_ds.class_names

print("\nClasses:", class_names)


# ============================================================
# 9. OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

print("Dataset pipeline optimized.")


# ============================================================
# 10. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)

print("Data augmentation created.")


# ============================================================
# 11. CALCULATE CLASS WEIGHTS
# ============================================================

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


# Create numerical labels based on the class counts.
labels = []

for class_index, class_name in enumerate(class_names):
    labels.extend(
        [class_index] * class_counts[class_name]
    )

labels = np.array(labels)

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


# ============================================================
# 12. BUILD CNN MODEL
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(224, 224, 3)
        ),

        data_augmentation,

        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(
            256,
            (3, 3),
            activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ]
)

model.summary()


# ============================================================
# 13. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.AUC(
            name="auc"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)

print("Model compiled successfully.")


# ============================================================
# 14. TRAINING CALLBACKS
# ============================================================

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


# ============================================================
# 15. TRAIN MODEL
# ============================================================

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


# ============================================================
# 16. PLOT TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 17. PLOT TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 18. EVALUATE MODEL ON TEST SET
# ============================================================

test_results = model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest results:")

for name, value in zip(
    model.metrics_names,
    test_results
):
    print(f"{name}: {value:.4f}")


# ============================================================
# 19. GENERATE PREDICTIONS
# ============================================================

y_true = []
y_prob = []

for images, labels_batch in test_ds:

    probabilities = model.predict(
        images,
        verbose=0
    )

    y_true.extend(
        labels_batch.numpy().flatten()
    )

    y_prob.extend(
        probabilities.flatten()
    )

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Convert probabilities into binary predictions.
y_pred = (y_prob >= 0.5).astype(int)

print("\nPredictions generated.")
print("Number of test images:", len(y_true))


# ============================================================
# 20. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot()

plt.title("Confusion Matrix")
plt.show()


# ============================================================
# 22. ROC-AUC
# ============================================================

auc_score = roc_auc_score(
    y_true,
    y_prob
)

print(f"\nROC-AUC: {auc_score:.4f}")


# ============================================================
# 23. SAVE TRAINED MODEL
# ============================================================

model_path = os.path.join(
    drive_folder,
    "pneumonia_cnn_model.keras"
)

model.save(model_path)

print("\nModel saved successfully!")
print(model_path)


# ============================================================
# 24. SAVE RESULTS
# ============================================================

results = {
    "test_accuracy": float(test_results[1]),
    "test_auc": float(test_results[2]),
    "test_precision": float(test_results[3]),
    "test_recall": float(test_results[4]),
    "roc_auc": float(auc_score)
}

results_path = os.path.join(
    drive_folder,
    "results.json"
)

with open(results_path, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("\nResults saved to:")
print(results_path)

print("\nFinal Results:")
print(json.dumps(results, indent=4))


# ============================================================
# 25. SHOW SAMPLE PREDICTIONS
# ============================================================

images, labels_batch = next(
    iter(test_ds)
)

probabilities = model.predict(
    images,
    verbose=0
).flatten()

plt.figure(figsize=(12, 10))

for i in range(min(9, len(images))):

    plt.subplot(3, 3, i + 1)

    plt.imshow(
        images[i].numpy().astype("uint8"),
        cmap="gray"
    )

    actual = class_names[
        int(labels_batch[i].numpy())
    ]

    predicted_index = int(
        probabilities[i] >= 0.5
    )

    predicted = class_names[
        predicted_index
    ]

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
