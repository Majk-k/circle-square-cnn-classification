# -*- coding: utf-8 -*-
"""
Circle vs. Square Classification with CNN

End-to-end script:
- downloads and extracts the dataset
- loads images and labels
- splits data into train/test sets
- trains two CNN models (with and without augmentation)
- plots training history
- evaluates models on the test set (confusion matrix + Cohen's Kappa)
"""

import os
import zipfile
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATASET_URL = "http://www.kasprowski.pl/datasets/circrec.zip"
ZIP_PATH = Path("circrec.zip")
EXTRACT_DIR = Path("circrec")
IMAGES_DIR = EXTRACT_DIR / "data"
FIGURES_DIR = Path("reports") / "figures"
MODELS_DIR = Path("models")


def download_and_extract_dataset() -> None:
    """Download and extract the dataset if not already present."""
    if not ZIP_PATH.exists():
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download completed.")
    else:
        print("ZIP file already exists, skipping download.")

    if not EXTRACT_DIR.exists():
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction completed.")
    else:
        print("Dataset directory already exists, skipping extraction.")


def load_images(indir: Path, size=(64, 64)):
    """
    Load images and labels from the given directory.
    File naming convention: img_<no>_<class>.jpg
    where <class> is 0 (square) or 1 (circle).
    """
    images = []
    labels = []

    for fname in os.listdir(indir):
        if not fname.lower().endswith(".jpg"):
            continue
        filepath = indir / fname

        img = cv2.imread(str(filepath))
        if img is None:
            print(f"Warning: could not read file {filepath}, skipping.")
            continue

        img = cv2.resize(img, size)
        images.append(img)

        cls_tag = fname.split("_")[-1].split(".")[0]
        try:
            label = int(cls_tag)
            if label not in (0, 1):
                raise ValueError
        except ValueError:
            print(f"Unexpected class tag '{cls_tag}' in file {fname}, assigning 0.")
            label = 0
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def build_cnn_model(input_shape=(64, 64, 3)) -> Sequential:
    """Build a simple CNN model for binary classification."""
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),

        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),

        Conv2D(128, 3, padding="same", activation="relu"),
        MaxPooling2D(),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_random_examples(images, labels):
    """Plot random examples from both classes."""
    idxs0 = np.where(labels == 0)[0]
    idxs1 = np.where(labels == 1)[0]

    if len(idxs0) < 3 or len(idxs1) < 3:
        print("Not enough samples to show random examples.")
        return

    sample0 = np.random.choice(idxs0, size=3, replace=False)
    sample1 = np.random.choice(idxs1, size=3, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for i, idx in enumerate(sample0):
        ax = axes[0, i]
        img = images[idx]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"Idx {idx} – Label: 0 (square)")
        ax.axis("off")

    for i, idx in enumerate(sample1):
        ax = axes[1, i]
        img = images[idx]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(f"Idx {idx} – Label: 1 (circle)")
        ax.axis("off")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "random_examples.png"
    plt.savefig(fig_path)
    print(f"Random examples saved to: {fig_path}")
    plt.show()


def train_models(X_train, y_train, X_test, y_test, batch_size=8, max_epochs=250):
    """Train two models: without and with augmentation."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_not_aug = build_cnn_model(input_shape=X_train.shape[1:])
    model_aug = build_cnn_model(input_shape=X_train.shape[1:])

    print("Model (no augmentation) summary:")
    model_not_aug.summary()
    print("\nModel (with augmentation) summary:")
    model_aug.summary()

    train_datagen_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_datagen_no_aug = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen_aug = train_datagen_aug.flow(X_train, y_train, batch_size=batch_size)
    train_gen_no_aug = train_datagen_no_aug.flow(X_train, y_train, batch_size=batch_size)
    val_gen = test_datagen.flow(X_test, y_test, batch_size=batch_size)

    # Visualize augmented vs non-augmented samples
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        batch_images, batch_labels = next(train_gen_aug)
        img_rgb = cv2.cvtColor((batch_images[0] * 255).astype("uint8"), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"Label: {int(batch_labels[0])}")
        plt.axis("off")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "augmented_samples.png"
    plt.savefig(fig_path)
    print(f"Augmented samples saved to: {fig_path}")
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        batch_images, batch_labels = next(train_gen_no_aug)
        img_rgb = cv2.cvtColor((batch_images[0] * 255).astype("uint8"), cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"Label: {int(batch_labels[0])}")
        plt.axis("off")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "non_augmented_samples.png"
    plt.savefig(fig_path)
    print(f"Non-augmented samples saved to: {fig_path}")
    plt.show()

    callbacks_no_aug = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ModelCheckpoint(
            MODELS_DIR / "best_model_not_augmented.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    callbacks_aug = [
        EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
        ModelCheckpoint(
            MODELS_DIR / "best_model_augmented.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("Training model without augmentation...")
    history_no_aug = model_not_aug.fit(
        train_gen_no_aug,
        epochs=max_epochs,
        validation_data=val_gen,
        callbacks=callbacks_no_aug,
        verbose=1,
    )

    print("Training model with augmentation...")
    history_aug = model_aug.fit(
        train_gen_aug,
        epochs=max_epochs,
        validation_data=val_gen,
        callbacks=callbacks_aug,
        verbose=1,
    )

    plot_training_history(history_no_aug, history_aug)

    return model_not_aug, model_aug


def plot_training_history(history_no_aug, history_aug):
    """Plot training history (loss and accuracy) for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(history_no_aug.history["loss"], label="train_loss_no_aug")
    axes[0, 0].plot(history_no_aug.history["val_loss"], label="val_loss_no_aug")
    axes[0, 0].set_title("Loss (no augmentation)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(history_aug.history["loss"], label="train_loss_aug")
    axes[0, 1].plot(history_aug.history["val_loss"], label="val_loss_aug")
    axes[0, 1].set_title("Loss (with augmentation)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()

    axes[1, 0].plot(history_no_aug.history["accuracy"], label="train_acc_no_aug")
    axes[1, 0].plot(history_no_aug.history["val_accuracy"], label="val_acc_no_aug")
    axes[1, 0].set_title("Accuracy (no augmentation)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()

    axes[1, 1].plot(history_aug.history["accuracy"], label="train_acc_aug")
    axes[1, 1].plot(history_aug.history["val_accuracy"], label="val_acc_aug")
    axes[1, 1].set_title("Accuracy (with augmentation)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / "training_history.png"
    plt.savefig(fig_path)
    print(f"Training history saved to: {fig_path}")
    plt.show()


def evaluate_models(X_test, y_test, model_no_aug, model_aug):
    """Evaluate both models on the test set."""
    X_test_norm = X_test.astype("float32") / 255.0

    models = {
        "No augmentation": model_no_aug,
        "With augmentation": model_aug,
    }

    for name, mdl in models.items():
        print(f"\nEvaluating model: {name}")
        y_pred_prob = mdl.predict(X_test_norm)
        y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
        y_true = y_test

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Square", "Circle"],
            yticklabels=["Square", "Circle"],
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix – {name}")
        plt.tight_layout()

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig_path = FIGURES_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fig_path)
        print(f"Confusion matrix for '{name}' saved to: {fig_path}")
        plt.show()

        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Cohen's Kappa ({name}): {kappa:.3f}")


def main():
    download_and_extract_dataset()

    images, labels = load_images(IMAGES_DIR)
    print(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")
    print("Class distribution:", Counter(labels))

    plot_random_examples(images, labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.5,
        stratify=labels,
        random_state=0,
    )

    print("\nAfter train/test split:")
    print(f"- Train: {X_train.shape[0]} images, class distribution: {Counter(y_train)}")
    print(f"- Test : {X_test.shape[0]} images, class distribution: {Counter(y_test)}")

    model_no_aug, model_aug = train_models(X_train, y_train, X_test, y_test)

    evaluate_models(X_test, y_test, model_no_aug, model_aug)


if __name__ == "__main__":
    main()
