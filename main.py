import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

# Load data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

processed.cleveland.data"

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 

 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv(url, names=cols)

# Preprocessing: Handle missing values and scale

df.replace('?', np.nan, inplace=True)

df.dropna(inplace=True)

X = df.drop('target', axis=1).values

y = (df['target'].values > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 

random_state=101)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# Build ANN

model = Sequential([

 Dense(64, activation='relu', input_shape=(X_train.shape[1],)),

 Dropout(0.2),

 Dense(32, activation='relu'),

 Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train

history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2, 

verbose=0)

# Evaluate

y_prob = model.predict(X_test)

y_pred = (y_prob > 0.5).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
