# ANN - MNIST Handwritten Digit Classification

An Artificial Neural Network (Multi-Layer Perceptron) built with TensorFlow and Keras that classifies handwritten digits from the MNIST dataset.

## About the Project

MNIST contains 70,000 grayscale images of handwritten digits (0 to 9), each 28x28 pixels. This project flattens each image into 784 values and passes them through fully connected layers to predict the digit.

## Features

- Data exploration and normalization of pixel values
- Fully connected layers with Batch Normalization, Dropout and L2 regularization
- Callbacks: EarlyStopping, ReduceLROnPlateau and ModelCheckpoint
- Training curves for accuracy and loss
- Evaluation with a classification report and confusion matrix
- Visualization of misclassified digits
- Saves the trained model and reloads it for a single prediction

## Model Architecture

| Layer | Details |
|-------|---------|
| Input | 784 values (flattened 28 x 28 image) |
| Hidden 1 | Dense (512) + BatchNorm + ReLU, Dropout 0.3 |
| Hidden 2 | Dense (256) + BatchNorm + ReLU, Dropout 0.3 |
| Hidden 3 | Dense (128) + BatchNorm + ReLU, Dropout 0.2 |
| Output | Dense (10) with softmax |

## Requirements

- Python 3.9 or higher
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn

Install everything with:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Usage

```bash
python ann_full.py
```

The MNIST dataset downloads automatically the first time you run the script.

## Output

All results are saved in the `ann_outputs/` folder:

| File | Description |
|------|-------------|
| `samples.png` | Sample digit images |
| `training_curves.png` | Accuracy and loss over epochs |
| `confusion_matrix.png` | Confusion matrix on the test set |
| `misclassified.png` | Examples the model got wrong |
| `best_ann.keras` | Best model during training |
| `ann_final.keras` | Final trained model |

## Results

Add your test accuracy here after running the script.

