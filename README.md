# CodeAlpha Machine Learning Internship — Task 3: Handwritten Character Recognition

## 📌 Project Overview
This repository contains the implementation for **Task 3** of the CodeAlpha Machine Learning Internship. The goal of this project is to build, train, and evaluate a **Convolutional Neural Network (CNN)** to classify handwritten characters/digits from image data locally in **VS Code**.

---

## 📂 Dataset & Preprocessing
* **Dataset:** MNIST Handwritten Digits (`28x28` grayscale images across 10 classes: `0–9`, with 60,000 training images and 10,000 test images).
* **Normalization:** Pixel values scaled from `[0, 255]` to `[0.0, 1.0]` (`float32`).
* **Channel Reshaping:** Expanded dimensions to `(samples, 28, 28, 1)` so convolutional layers can process single-channel spatial features.

---

## 🛠️ Tech Stack & Dependencies
* **Language:** Python 3.x
* **Development Environment:** VS Code (Local Workspace)
* **Libraries Used:**
  * `tensorflow` / `keras` — CNN architecture, training, and model persistence
  * `numpy` — Array manipulation and label decoding
  * `scikit-learn` — Confusion matrix and classification report metrics
  * `matplotlib` & `seaborn` — Training curves, sample previews, and confusion matrix heatmap

---

## 🧠 CNN Architecture & Training
* **Convolutional & Pooling Blocks:**
  * `Conv2D` (32 filters, `3x3`, ReLU, `padding='same'`) + `MaxPooling2D(2, 2)`
  * `Conv2D` (64 filters, `3x3`, ReLU, `padding='same'`) + `MaxPooling2D(2, 2)`
  * `Conv2D` (128 filters, `3x3`, ReLU, `padding='same'`)
* **Classification Head:** `Flatten` → `Dropout(0.3)` → `Dense(128, relu)` → `Dense(10, softmax)`
* **Compilation & Training:** Trained with the `adam` optimizer, `sparse_categorical_crossentropy` loss, `batch_size=128`, `validation_split=0.1`, and `epochs=5`, then saved to `handwritten_cnn_model.keras`.

---

## 📊 Evaluation & Visualizations
* **Test Metrics:** Overall test accuracy, test loss, and a per-class classification report (precision, recall, and F1-score).
* **Visual Outputs:**
  * Sample handwritten input images (`2x5` grid)
  * Training vs. validation accuracy and loss curves
  * `10x10` confusion matrix heatmap
  * Test set prediction grid highlighting correct and misclassified samples

---

## 🚀 How to Run the Project Locally

1. Clone or download this repository and open the folder in **VS Code**.
2. Install the required dependencies via the VS Code terminal:
   ```bash
   pip install -r requirements.txt


## 👤 Author
* **Priyanshi Rastogi**  
* *B.Tech Computer Science and Engineering, IIIT Pune*  
* *CodeAlpha Machine Learning Intern*
