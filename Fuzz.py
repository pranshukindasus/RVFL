import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report

class RVFLFuzzyClassifier:
    def __init__(self, n_hidden_units=100, alpha=1.0):
        self.n_hidden_units = n_hidden_units
        self.alpha = alpha
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_model = RidgeClassifier(alpha=alpha)

    def fuzzy_activation(self, X):
        # Example of a fuzzy activation function (triangular fuzzy function)
        return np.maximum(0, 1 - np.abs(X))

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape

        # Randomly initialize hidden layer weights and biases
        self.hidden_weights = np.random.randn(n_features, self.n_hidden_units)
        self.hidden_bias = np.random.randn(self.n_hidden_units)

        # Compute hidden layer output with fuzzy activation
        H = self.fuzzy_activation(X.dot(self.hidden_weights) + self.hidden_bias)

        # Concatenate original features and hidden layer output
        H_ext = np.hstack((X, H))

        # Train the output layer model
        self.output_model.fit(H_ext, y)

    def predict(self, X):
        H = self.fuzzy_activation(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X, H))
        return self.output_model.predict(H_ext)

    def predict_proba(self, X):
        H = self.fuzzy_activation(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X, H))
        return self.output_model.decision_function(H_ext)

# Load the training dataset (replace 'path_to_train_dataset.csv' with the actual path)
train_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\FilteredTrain.csv', encoding='latin1')

# Separate features and labels for training data
X_train = train_data.iloc[:, 3:]  # Exclude the first three columns (Serial Number, Chemical Structure, Label)
y_train = train_data.iloc[:, 2]   # Use the third column as the label

# Load the test dataset (replace 'path_to_test_dataset.csv' with the actual path)
test_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\FilteredTest.csv', encoding='latin1')

# Separate features and labels for test data
X_test = test_data.iloc[:, 3:]  # Exclude the first three columns (Serial Number, Chemical Structure, Label)
y_test = test_data.iloc[:, 2]   # Use the third column as the label

# Convert feature columns to numeric types
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values resulting from conversion errors
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Align y_train with X_train
X_test = X_test.dropna()
y_test = y_test[X_test.index]  # Align y_test with X_test

# Instantiate the RVFLFuzzyClassifier model
rvfl_fuzzy = RVFLFuzzyClassifier(n_hidden_units=100, alpha=1.0)

# Train the model
rvfl_fuzzy.fit(X_train, y_train)

# Predict labels for the test set
predictions_fuzzy = rvfl_fuzzy.predict(X_test)

results_df = pd.DataFrame({'Actual_Label': y_test, 'Predicted_Label': predictions_fuzzy})

# Save the results to a CSV file
results_df.to_csv('test_predictionsfuzz.csv', index=False)

print("Test predictions saved to 'test_predictions.csv'")

# Evaluate the model with fuzzy activation
print("Accuracy (with fuzzy activation):", accuracy_score(y_test, predictions_fuzzy))
print("Classification Report (with fuzzy activation):\n", classification_report(y_test, predictions_fuzzy))
