import numpy as np
import pandas as pd
import sklearn

# Load the training dataset
train_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\FilteredTrain.csv', encoding='latin1')
test_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\FilteredTest.csv', encoding='latin1')

# Separate features and labels
X_train = train_data.iloc[:, 3:]
y_train = train_data.iloc[:, 2]
X_test = test_data.iloc[:, 3:]
y_test = test_data.iloc[:, 2]

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier

class RVFLClassifier:
    def __init__(self, n_hidden_units=100, alpha=1.0):
        self.n_hidden_units = n_hidden_units
        self.alpha = alpha
        self.hidden_weights = None
        self.hidden_bias = None
        self.output_model = RidgeClassifier(alpha=alpha)

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape

        # Randomly initialize hidden layer weights and biases
        self.hidden_weights = np.random.randn(n_features, self.n_hidden_units)
        self.hidden_bias = np.random.randn(self.n_hidden_units)

        # Compute hidden layer output
        H = np.tanh(X.dot(self.hidden_weights) + self.hidden_bias)

        # Concatenate original features and hidden layer output
        H_ext = np.hstack((X, H))

        # Train the output layer model
        self.output_model.fit(H_ext, y)

    def predict(self, X):
        H = np.tanh(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X, H))
        return self.output_model.predict(H_ext)

    def predict_proba(self, X):
        H = np.tanh(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X, H))
        return self.output_model.decision_function(H_ext)

# Instantiate the RVFL model
rvfl = RVFLClassifier(n_hidden_units=100, alpha=1.0)
rvfl.fit(X_train, y_train)
predictions = rvfl.predict(X_test)


test_predictions = rvfl.predict(X_test)

# Create a DataFrame to store the results
results_df = pd.DataFrame({'Actual_Label': y_test, 'Predicted_Label': test_predictions})

# Save the results to a CSV file
results_df.to_csv('test_predictions.csv', index=False)

print("Test predictions saved to 'test_predictions.csv'")
``
# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
