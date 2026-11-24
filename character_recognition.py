# ========================================================
# CodeAlpha Machine Learning Internship
# Task 3: Handwritten Character Recognition (CNN in VS Code)
# ========================================================
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# ------------------------------------------
# 1. LOAD & PREPROCESS DATASET (MNIST)
# ------------------------------------------
print("Loading handwritten dataset...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalize pixel values from [0, 255] to [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
# Reshape to add channel dimension: (samples, height, width, channels)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
print(f"Testing data shape : {X_test.shape}, Labels: {y_test.shape}")
# Preview sample training images
plt.figure(figsize=(8, 4))
for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(X_train[i].squeeze(), cmap="gray")
  plt.title(f"Label: {y_train[i]}")
  plt.axis("off")
plt.suptitle("Sample Handwritten Characters")
plt.tight_layout()
plt.show()
# ------------------------------------------
# 2. BUILD CNN MODEL
# ------------------------------------------
model = models.Sequential([
    layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="same"
    ),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()
# ------------------------------------------
# 3. TRAIN MODEL
# ------------------------------------------
print("\nTraining CNN model...")
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
)
# ------------------------------------------
# 4. EVALUATE ON TEST SET
# ------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")
# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# ------------------------------------------
# 5. VISUALIZATIONS (ACCURACY, LOSS & CONFUSION MATRIX)
# ------------------------------------------
# Training vs Validation Accuracy & Loss
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Handwritten Character Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
# Sample Test Predictions
plt.figure(figsize=(8, 4))
for i in range(10):
  plt.subplot(2, 5, i + 1)
  plt.imshow(X_test[i].squeeze(), cmap="gray")
  color = "green" if y_pred[i] == y_test[i] else "red"
  plt.title(f"Pred: {y_pred[i]} (True: {y_test[i]})", color=color, fontsize=9)
  plt.axis("off")
plt.suptitle("Test Predictions (Green = Correct, Red = Incorrect)")
plt.tight_layout()
plt.show()
# Save trained model
model.save("handwritten_cnn_model.keras")
print("Saved trained model to 'handwritten_cnn_model.keras'")