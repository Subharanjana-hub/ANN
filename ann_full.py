"""
ANN (Artificial Neural Network / MLP) - Full Pipeline
Dataset : MNIST handwritten digits
Features: data exploration, preprocessing, BatchNorm + Dropout, callbacks
          (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint), training curves,
          confusion matrix, classification report, model saving, sample predictions

Install : pip install tensorflow numpy matplotlib scikit-learn
Run     : python ann_full.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
OUT_DIR = "ann_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
print("Train shape:", x_train_full.shape, "| Test shape:", x_test.shape)
print("Classes    :", np.unique(y_train_full))
print("Class counts (train):", np.bincount(y_train_full))

x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train_full = x_train_full.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

val_size = 6000
x_val, y_val = x_train_full[:val_size], y_train_full[:val_size]
x_train, y_train = x_train_full[val_size:], y_train_full[val_size:]
print(f"Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test_flat.shape}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(x_test[i], cmap="gray")
    ax.set_title(f"Label: {y_test[i]}")
    ax.axis("off")
plt.suptitle("Sample MNIST images")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/samples.png", dpi=120)
plt.close()

def build_ann(input_dim=784, num_classes=10):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(512, kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(256, kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ], name="ann_mnist")
    return model

model = build_ann()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.ModelCheckpoint(f"{OUT_DIR}/best_ann.keras",
                                    monitor="val_accuracy",
                                    save_best_only=True, verbose=0),
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=128,
    callbacks=callbacks,
    verbose=2,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy"); ax[0].set_xlabel("Epoch"); ax[0].legend()

ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

y_prob = model.predict(x_test_flat, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
for i in range(10):
    for j in range(10):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

wrong = np.where(y_pred != y_test)[0]
print(f"\nTotal misclassified: {len(wrong)} / {len(y_test)}")
fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
for ax, idx in zip(axes.ravel(), wrong[:10]):
    ax.imshow(x_test[idx], cmap="gray")
    ax.set_title(f"True {y_test[idx]} | Pred {y_pred[idx]}", color="red", fontsize=9)
    ax.axis("off")
plt.suptitle("Misclassified examples")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/misclassified.png", dpi=120)
plt.close()

model.save(f"{OUT_DIR}/ann_final.keras")
loaded = keras.models.load_model(f"{OUT_DIR}/ann_final.keras")

sample = x_test_flat[0:1]
probs = loaded.predict(sample, verbose=0)[0]
print(f"\nSingle prediction -> digit {np.argmax(probs)} "
      f"(confidence {probs.max():.2%}), true label {y_test[0]}")
print(f"\nAll outputs saved in: {OUT_DIR}/")
