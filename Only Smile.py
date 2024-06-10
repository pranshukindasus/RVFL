import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
        H_ext = np.hstack((X.toarray(), H))

        # Train the output layer model
        self.output_model.fit(H_ext, y)

    def predict(self, X):
        H = self.fuzzy_activation(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X.toarray(), H))
        return self.output_model.predict(H_ext)

    def predict_proba(self, X):
        H = self.fuzzy_activation(X.dot(self.hidden_weights) + self.hidden_bias)
        H_ext = np.hstack((X.toarray(), H))
        return self.output_model.decision_function(H_ext)

# Load the training dataset (replace 'path_to_train_dataset.csv' with your actual training dataset path)
train_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\OnlySMILE_Train.csv', encoding='latin1')

# Separate features and labels for training data
X_train = train_data['SMILE ID']
y_train = train_data['LABEL']

# Convert labels to numeric values
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Vectorize the compound formula using TF-IDF
tfidf = TfidfVectorizer()
X_train_vectorized = tfidf.fit_transform(X_train)

# Instantiate the RVFLFuzzyClassifier model
rvfl_fuzzy = RVFLFuzzyClassifier(n_hidden_units=100, alpha=1.0)

# Train the model
rvfl_fuzzy.fit(X_train_vectorized, y_train_encoded)

# Load the test dataset (replace 'path_to_test_dataset.csv' with your actual test dataset path)
test_data = pd.read_csv(r'C:\Users\Lenovo\Desktop\Proj Work\TEST\OnlySMILE_Test.csv', encoding='latin1')

# Separate features and labels for test data
X_test = test_data['SMILE ID']
y_test = test_data['LABEL']

# Convert labels to numeric values for the test set
y_test_encoded = label_encoder.transform(y_test)

# Vectorize the test compound formula using the same TF-IDF vectorizer
X_test_vectorized = tfidf.transform(X_test)

# Predict labels for the test set
predictions_fuzzy = rvfl_fuzzy.predict(X_test_vectorized)

# Evaluate the model with fuzzy activation
print("Accuracy (with fuzzy activation):", accuracy_score(y_test_encoded, predictions_fuzzy))
print("Classification Report (with fuzzy activation):\n", classification_report(y_test_encoded, predictions_fuzzy))

# Save predictions and actual labels to a CSV file
results_df = pd.DataFrame({'Actual_Label': label_encoder.inverse_transform(y_test_encoded), 'Predicted_Label': label_encoder.inverse_transform(predictions_fuzzy)})
results_df.to_csv('OnlySmilePred.csv', index=False)
print("Test predictions saved to 'OnlySmilePred.csv'")

