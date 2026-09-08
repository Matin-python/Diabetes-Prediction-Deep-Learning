import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv('diabetes2.csv')

print(df.head(10))
print(df.describe())

# Separate features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential()

model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(1))


# Compile the model
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)


# Display model architecture
model.summary()


# Train the model
h = model.fit(
    X_train,
    y_train,
    epochs=200
)


# Make predictions
y_pred_probability = model.predict(X_test)

# Convert probabilities to 0 or 1
y_pred = (y_pred_probability >= 0.5).astype(int).flatten()

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

print("=" * 60)
print("Accuracy =", accuracy * 100, "%")

# Calculate Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)

print("=" * 50)
print("Mean Squared Error =", mse)
print("Final Training Loss =", h.history['loss'][-1])
print("=" * 50)


# Plot training loss
loss = h.history['loss']

plt.close()
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()