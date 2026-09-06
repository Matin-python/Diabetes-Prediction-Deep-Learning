# 🩺 Diabetes Prediction using Deep Learning

A Deep Learning project that uses an Artificial Neural Network (ANN) to predict whether a patient has diabetes based on medical diagnostic features.

## Overview

This project uses a diabetes dataset to build a binary classification model using an Artificial Neural Network.

The model receives **8 medical features** as input and predicts the `Outcome` of the patient:

* `0` → No Diabetes
* `1` → Diabetes

Before training the neural network, the input features are standardized using `StandardScaler` to improve the training process.

The project also includes dataset exploration, data preprocessing, model training, prediction, accuracy evaluation, Mean Squared Error calculation, and training loss visualization.

## Features

* 🩺 Diabetes prediction
* 🤖 Artificial Neural Network
* 📊 Binary classification
* 🔄 Feature standardization
* 📈 Model training
* 🎯 Prediction on test data
* ✅ Accuracy calculation
* 📉 Mean Squared Error calculation
* 📊 Training loss visualization
* 🧠 TensorFlow/Keras implementation

## Technologies Used

* Python
* NumPy
* Pandas
* TensorFlow
* Keras
* Scikit-learn
* Matplotlib

## Dataset

The project uses a diabetes dataset containing medical diagnostic measurements.

The dataset contains **8 input features**:

* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`

The target variable is:

* `Outcome`

where:

```text
0 = No Diabetes
1 = Diabetes
```

## Data Preprocessing

The dataset is loaded using Pandas:

```python
df = pd.read_csv('diabetes2.csv')
```

The input features and target variable are separated:

```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

The dataset is divided into training and testing sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Feature Standardization

The input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then applied to the test data.

This helps put the features on a similar scale and can improve neural network training.

## Deep Learning Model

The project uses a simple Artificial Neural Network built with Keras.

The model architecture is:

```text
8 Input Features
       ↓
Dense Layer
16 Neurons
ReLU Activation
       ↓
Dense Layer
1 Neuron
       ↓
Prediction
```

The model is created using:

```python
model = Sequential()

model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dense(1))
```

The model contains:

* **8 input features**
* **16 neurons** in the hidden layer
* **ReLU activation** in the hidden layer
* **1 output neuron**

## Model Compilation

The model is compiled using the Adam optimizer and Mean Squared Error loss:

```python
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
```

## Model Training

The neural network is trained using the training dataset for 200 epochs:

```python
h = model.fit(
    X_train,
    y_train,
    epochs=200
)
```

During training, the model adjusts its weights to reduce the loss between its predictions and the actual target values.

## Prediction

After training, the model makes predictions using the test dataset:

```python
y_pred_probability = model.predict(X_test)
```

The model output is then converted into binary predictions using a threshold of `0.5`:

```python
y_pred = (y_pred_probability >= 0.5).astype(int).flatten()
```

Therefore:

```text
Prediction >= 0.5 → 1
Prediction < 0.5  → 0
```

## Evaluation

The model is evaluated using Accuracy and Mean Squared Error.

### Accuracy

Accuracy is calculated by comparing the predicted output with the real output:

```python
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy =", accuracy * 100, "%")
```

The accuracy represents the percentage of test samples that were classified correctly.

### Mean Squared Error

The project also calculates Mean Squared Error:

```python
mse = metrics.mean_squared_error(y_test, y_pred)

print("Mean Squared Error =", mse)
```

The Mean Squared Error measures the average squared difference between the predicted and actual values.

## Training Loss

The training loss recorded during model training is stored in:

```python
h.history['loss']
```

The loss is visualized using Matplotlib:

```python
loss = h.history['loss']

plt.close()
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
```

This graph shows how the training loss changes over the 200 training epochs.

## Example Output

The program displays information about the dataset and model and produces results similar to:

```text
============================================================
Accuracy = XX.XX %
==================================================
Mean Squared Error = X.XX
Final Training Loss = X.XX
==================================================
```

The exact accuracy and loss can vary depending on the dataset and model training.

## Project Structure

```text
Diabetes-prediction-Deep_Learning/
│
├── diabetes2.csv
├── diabetes_prediction.py
├── diabetes_prediction.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/Diabetes-prediction-Deep_Learning.git
```

Go to the project directory:

```bash
cd Diabetes-prediction-Deep_Learning
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Run the Python script:

```bash
python diabetes_prediction.py
```

The program will:

1. Load the diabetes dataset.
2. Display the first 10 rows and statistical information.
3. Separate the features and target.
4. Split the dataset into training and testing sets.
5. Standardize the input features.
6. Build the neural network.
7. Display the model architecture.
8. Train the model for 200 epochs.
9. Make predictions on the test dataset.
10. Convert predictions into `0` or `1`.
11. Calculate the accuracy.
12. Calculate Mean Squared Error.
13. Display the final training loss.
14. Plot the training loss.

The project can also be explored using the Jupyter Notebook.

```bash
jupyter notebook diabetes_prediction.ipynb
```

## 🔗 Related Project

This project is the **Deep Learning implementation** of the diabetes prediction problem.

For the Machine Learning implementation using Logistic Regression, see:

**Diabetes Prediction using Logistic Regression**

https://github.com/Matin-python/Diabetes-Prediction-Logistic-Regression

The related project uses **Logistic Regression** from Scikit-learn to solve the same binary classification problem.

While this repository uses an **Artificial Neural Network**, the related project uses a traditional Machine Learning approach.

### 🆚 Machine Learning vs Deep Learning

| Aspect           | Machine Learning      | Deep Learning             |
| ---------------- | --------------------- | ------------------------- |
| Approach         | Logistic Regression   | Artificial Neural Network |
| Type             | Supervised Learning   | Supervised Learning       |
| Problem          | Binary Classification | Binary Classification     |
| Target           | `Outcome`             | `Outcome`                 |
| Framework        | Scikit-learn          | TensorFlow / Keras        |
| Model complexity | Relatively simple     | More complex              |

These two projects demonstrate two different approaches to the diabetes prediction problem.

## Future Improvements

* 📊 Add Precision, Recall, and F1-score
* 📉 Add a confusion matrix
* 📈 Add ROC curve and AUC
* 🧠 Experiment with different neural network architectures
* ⚙️ Tune hyperparameters
* 🔄 Compare different activation functions
* 📊 Compare the ANN with Logistic Regression
* 🧪 Use cross-validation
* 💾 Save the trained model
* 🌐 Deploy the model as a web application

## Contributing

Contributions, suggestions, and bug reports are welcome.

Feel free to fork this repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Author

**Mohammad Reza Bakhshandeh**

Interested in Python, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
