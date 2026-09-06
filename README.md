# 🩺 Diabetes Prediction using Deep Learning

A deep learning classification project that uses an Artificial Neural Network (ANN) to predict whether a patient has diabetes based on medical diagnostic measurements.

## Overview

This project uses the Diabetes dataset to build a binary classification model using Deep Learning.

The target variable, `Outcome`, indicates whether a patient has diabetes:

* `0` → No diabetes
* `1` → Diabetes

An Artificial Neural Network (ANN) is built using TensorFlow/Keras and trained to learn the relationship between the medical features and the diabetes outcome.

The project also includes data preprocessing, train/test splitting, model training, prediction, and evaluation.

## Features

* 🩺 Diabetes prediction using Deep Learning
* 🤖 Binary classification
* 🧠 Artificial Neural Network (ANN)
* 📊 Train/test data splitting
* 📈 Model training and evaluation
* 📉 Mean Squared Error (MSE) calculation
* 🐼 Dataset handling using Pandas
* 🔥 Neural network model built with TensorFlow/Keras
* 📚 Beginner-friendly Deep Learning implementation

## Technologies Used

* Python 3
* NumPy
* Pandas
* TensorFlow
* Keras
* Scikit-learn
* Matplotlib

## Dataset

The project uses a diabetes dataset containing medical diagnostic measurements used to predict whether a patient has diabetes.

The dataset contains the following features:

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

## Data Preparation

The dataset is loaded using Pandas.

The input features are separated from the target variable:

```python
X = df.drop('Outcome', axis=1)
y = df.Outcome
```

The dataset is then divided into training and testing sets using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)
```

The training data is used to train the neural network, while the test data is used to evaluate its performance on unseen samples.

## Model

The project uses an Artificial Neural Network built with TensorFlow/Keras.

The model contains a dense hidden layer with ReLU activation and an output layer for the binary classification task.

```python
model = Sequential()

model.add(Dense(20, input_dim=8, activation='relu'))
model.add(Dense(1))
```

The model is compiled using the Adam optimizer and Mean Squared Error loss:

```python
model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=['mean_squared_error']
)
```

The model is then trained for multiple epochs:

```python
h = model.fit(
    X_train,
    y_train,
    epochs=200
)
```

## Model Architecture

The neural network can be represented as:

```text
Input Features
      │
      ▼
Dense Layer
20 Neurons
ReLU Activation
      │
      ▼
Output Layer
1 Neuron
      │
      ▼
Diabetes Prediction
```

The input layer receives the medical diagnostic features.

The hidden layer contains 20 neurons and uses the ReLU activation function to learn non-linear relationships between the input features.

The final layer produces the model's output for the diabetes prediction.

## Model Training

During training, the neural network learns from the training dataset by adjusting its weights to minimize the loss function.

The training process is performed for 200 epochs.

The training history is stored in the variable `h`, which can be used to examine how the loss changes during training.

```python
h = model.fit(X_train, y_train, epochs=200)
```

The training loss can then be visualized to observe the learning process.

## Evaluation

After training, the model is used to make predictions on the test dataset:

```python
y_pred = model.predict(X_test)
```

The project calculates Mean Squared Error to evaluate the difference between the predicted and actual values:

```python
metrics.mean_squared_error(y_test, y_pred)
```

Mean Squared Error measures the average squared difference between the predicted values and the actual target values.

For a binary classification problem, additional classification metrics such as Accuracy, Precision, Recall, F1-score, and ROC-AUC can also be used in future improvements.

## Training Loss

The training history can be used to visualize the model's loss during training.

```python
er = h.history['loss']
er = er[5:]

plt.close()
plt.plot(er)
plt.show()
```

This visualization helps show how the model's loss changes as the number of training epochs increases.

## Example Output

```text
Model: "sequential"

_______________________________________________________________
 Layer (type)                 Output Shape              Param #
=================================================================
 dense (Dense)                (None, 20)                ...
 dense_1 (Dense)              (None, 1)                 ...
=================================================================

Train the model...
Epoch 1/200
...
Epoch 200/200
...

Mean Squared Error = X.XX
Final Training Loss = X.XX
```

The exact results may vary between runs because the dataset is randomly divided into training and testing sets.

## Project Structure

```text
Diabetes-Prediction-Deep_Learning/
│
├── diabetes_prediction.py
├── diabetes_prediction.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Matin-python/Diabetes-Prediction-Deep_Learning.git
```

Move into the project directory:

```bash
cd Diabetes-Prediction-Deep_Learning
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy pandas tensorflow keras scikit-learn matplotlib
```

## How to Run

Run the Python script:

```bash
python diabetes_prediction.py
```

The program will:

* Load the diabetes dataset.
* Display basic information about the dataset.
* Separate the input features and target variable.
* Split the dataset into training and testing sets.
* Build the Artificial Neural Network.
* Display the model architecture.
* Train the neural network.
* Make predictions on the test dataset.
* Calculate Mean Squared Error.
* Display the training loss.

The project can also be explored using the Jupyter Notebook:

```bash
jupyter notebook diabetes_prediction.ipynb
```

## Related Project

This project uses an Artificial Neural Network built with TensorFlow/Keras to solve the diabetes prediction problem.

For a Logistic Regression implementation of the same problem, see:

### 🧠 Diabetes Prediction using Logistic Regression

This project uses Logistic Regression from Scikit-learn to solve the diabetes binary classification problem.

### Comparison

| Aspect           | Machine Learning      | Deep Learning             |
| ---------------- | --------------------- | ------------------------- |
| Approach         | Logistic Regression   | Artificial Neural Network |
| Type             | Supervised Learning   | Supervised Learning       |
| Problem          | Binary Classification | Binary Classification     |
| Target           | `Outcome`             | `Outcome`                 |
| Framework        | Scikit-learn          | TensorFlow / Keras        |
| Model complexity | Relatively simple     | More complex              |
| Learning method  | Logistic Regression   | Neural Network            |

These two projects demonstrate how the same type of classification problem can be approached using both traditional Machine Learning and Deep Learning techniques.

➡️ Repository: https://github.com/Matin-python/Diabetes-Prediction-Logistic-Regression

## License

This project is licensed under the MIT License.

## Author

Mohammad Reza Bakhshandeh

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
