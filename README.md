# Heart Disease Classification using Deep Learning ❤️🩺

## Overview
This project implements an Artificial Neural Network (ANN) using TensorFlow and Keras to predict the presence of heart disease in patients. It was developed as a mini-project to demonstrate the end-to-end deep learning pipeline, from data preprocessing to model evaluation.

**Developer:** Aritraa

## Dataset
This project uses the standard **Heart Disease Dataset** from the UCI Machine Learning Repository.
* **Instances:** 303
* **Features:** 13 clinical attributes (e.g., age, cholesterol, maximum heart rate)
* **Target:** Binary classification (0 = No Disease, 1 = Disease Present)

## Tech Stack
* **Language:** Python
* **Libraries:** TensorFlow/Keras, Scikit-learn, Pandas, NumPy, Matplotlib

## Model Architecture
The network is a Sequential ANN built to handle tabular numerical data:
* **Input Layer:** Standardized 13-feature input
* **Hidden Layers:** * Dense (64 neurons, ReLU activation)
  * Dropout (0.2) to mitigate overfitting
  * Dense (32 neurons, ReLU activation)
* **Output Layer:** Dense (1 neuron, Sigmoid activation for binary classification)
* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy

## Performance Metrics (Test Set)
The model generalizes well to unseen data, achieving the following metrics:
* **Accuracy:** 85.0%
* **Precision:** 84.6%
* **Recall:** 81.4%
* **F1-Score:** 82.9%
* **AUC Score:** 0.91

## How to Run
1. Clone the repository: `git clone https://github.com/your-username/heart-disease-deep-learning.git`
2. Ensure you have the required libraries installed: `pip install tensorflow pandas numpy matplotlib scikit-learn`
3. Run the Jupyter Notebook `heart_disease_classification.ipynb` to view the training process and evaluation graphs.
