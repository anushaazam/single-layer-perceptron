# Single-Layer Perceptron for Banknote Authentication

## Project Overview

This project implements a **Single-Layer Perceptron from scratch using Python and NumPy** to classify banknotes as genuine or forged.

The Banknote Authentication dataset is used to train and evaluate the perceptron model. The implementation demonstrates the fundamental concepts of an Artificial Neural Network (ANN), including weights, bias, learning rate, activation function, training, prediction, and model evaluation.

The model is developed without using pre-built machine learning classifiers. The perceptron learning algorithm is implemented manually using NumPy.

## Project Objectives

- Understand the working principle of a single-layer perceptron.
- Load and explore the Banknote Authentication dataset.
- Perform data preprocessing and feature standardization.
- Implement the perceptron learning algorithm from scratch.
- Train the model using a training dataset.
- Predict whether a banknote is genuine or forged.
- Evaluate the model using classification metrics.
- Visualize the training progress using a misclassification plot.

## Dataset

This project uses the **Banknote Authentication Dataset**.

**Dataset Source:**  
[Kaggle – Bank Note Authentication UCI Data](https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data)

**Dataset File:** `BankNote_Authentication.csv`

The dataset contains numerical features extracted from images of genuine and forged banknotes using a wavelet transformation.

### Dataset Features

| Feature | Description |
|---|---|
| Variance | Variance of the wavelet-transformed image |
| Skewness | Skewness of the wavelet-transformed image |
| Curtosis | Curtosis of the wavelet-transformed image |
| Entropy | Entropy of the image |
| Class | Target classification label |

### Target Classes

- `0` – Genuine banknote
- `1` – Forged banknote

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Google Colab / Jupyter Notebook

## Implementation

### 1. Dataset Loading

The dataset is loaded using Pandas.

```python
data = pd.read_csv("BankNote_Authentication.csv")
```

The dataset shape, first few records, missing values, duplicate rows, and class distribution are examined.

### 2. Data Preprocessing

The following preprocessing steps are performed:

- Check for missing values.
- Identify and remove duplicate records.
- Separate input features (X) and target labels (y).
- Shuffle the dataset using a fixed random seed.
- Split the dataset into training and testing sets.
- Standardize the features using the training set's mean and standard deviation.

### 3. Train-Test Split

The dataset is divided into:

- 80% training data.
- 20% testing data.

The training data is used to learn the perceptron weights and bias, while the testing data is used to evaluate its performance.

### 4. Perceptron Implementation

The perceptron is implemented from scratch using a Python class.

**Model Parameters:**

| Parameter | Value |
|---|---|
| Learning rate | 0.01 |
| Number of epochs | 100 |
| Initial weights | Zeros |
| Initial bias | 0 |
| Activation function | Step function |

### 5. Activation Function

The step activation function is used to convert the weighted input into a binary prediction.

The mathematical representation is:

\[
f(z)=
\begin{cases}
1, & z \geq 0\\
0, & z < 0
\end{cases}
\]

Where:

\[
z = \mathbf{w}\cdot\mathbf{x}+b
\]

Here:
- \(w\) represents the weights.
- \(x\) represents the input features.
- \(b\) represents the bias.

### 6. Weight and Bias Update

The perceptron learning rule is used to update the weights and bias during training.

\[
\text{Update} = \text{Learning Rate}\times(\text{Actual}-\text{Predicted})
\]

\[
w_{\text{new}}=w_{\text{old}}+\text{Update}\times x
\]

\[
b_{\text{new}}=b_{\text{old}}+\text{Update}
\]

The training process continues for 100 epochs.

### 7. Model Evaluation

After training, the model predicts the classes of the testing data.

The following evaluation metrics are calculated manually:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 8. Training Visualization

Matplotlib is used to visualize the number of misclassifications in each training epoch.

The graph shows how the perceptron's training errors change throughout the training process.

## Project Structure

```text
single-layer-perceptron/
│
├── BankNote_Authentication.csv
├── your_notebook.ipynb
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/anushaazam/single-layer-perceptron.git
```

### 2. Open the Project Directory

```bash
cd single-layer-perceptron
```

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib
```

### 4. Open the Notebook

Open the `.ipynb` file using Google Colab or Jupyter Notebook.

### 5. Load the Dataset

Make sure `BankNote_Authentication.csv` is available in the notebook's working directory.

### 6. Run the Notebook

Execute all notebook cells in order to preprocess the dataset, train the perceptron, evaluate predictions, and display the training progress graph.

## Conclusion

This project demonstrates the implementation of a Single-Layer Perceptron from scratch for binary classification using the Banknote Authentication dataset.

It provides practical experience with the basic architecture of an artificial neural network, including the perceptron learning algorithm, step activation function, weight updates, bias adjustment, and model evaluation.

The project also demonstrates how NumPy can be used to build a simple neural network without relying on pre-built machine learning classification models.

## Author

**Anusha Azam**

This project is intended for educational and academic purposes.
