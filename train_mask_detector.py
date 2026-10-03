"""Train a MobileNetV2-based face mask classifier.

Expected dataset layout:
    dataset/
        with_mask/      *.jpg / *.png
        without_mask/   *.jpg / *.png

Usage:
    python train_mask_detector.py --dataset dataset --epochs 10
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (224, 224)


def parse_args():
    ap = argparse.ArgumentParser(description="Train the face mask classifier")
    ap.add_argument("--dataset", default="dataset", help="path to dataset folder")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model", default="mask_detector.keras", help="output model path")
    ap.add_argument("--labels", default="labels.json", help="output class-names file")
    ap.add_argument("--plot", default="training_plot.png", help="output plot path")
    return ap.parse_args()


def load_datasets(path, batch_size):
    common = dict(
        validation_split=0.2,
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        path, subset="training", **common
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        path, subset="validation", **common
    )
    return train_ds, val_ds, train_ds.class_names


def build_model():
    base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False  # freeze the pretrained base

    x = AveragePooling2D(pool_size=(7, 7))(base.output)
    x = Flatten(name="flatten")(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation="softmax")(x)
    return Model(inputs=base.input, outputs=x)


def main():
    args = parse_args()

    train_ds, val_ds, class_names = load_datasets(args.dataset, args.batch_size)
    print(f"[INFO] Classes (index order): {class_names}")

    augment = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ]
    )

    autotune = tf.data.AUTOTUNE
    train_ds = (
        train_ds.map(
            lambda x, y: (preprocess_input(augment(x, training=True)), y),
            num_parallel_calls=autotune,
        )
        .prefetch(autotune)
    )
    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=autotune
    ).prefetch(autotune)

    model = build_model()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=args.lr),
        metrics=["accuracy"],
    )

    print("[INFO] Training head...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    print("[INFO] Evaluating on validation set...")
    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    print(classification_report(y_true, y_pred, target_names=class_names))

    model.save(args.model)
    with open(args.labels, "w") as f:
        json.dump(class_names, f)
    print(f"[INFO] Saved model to {args.model} and labels to {args.labels}")

    plt.figure()
    plt.plot(history.history["accuracy"], label="train acc")
    plt.plot(history.history["val_accuracy"], label="val acc")
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.legend(loc="best")
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
