import pandas as pd
import numpy as np
from sklearn import ensemble
import bentoml

# Load data
X_train = pd.read_csv('data/preprocessed/X_train.csv')
X_test = pd.read_csv('data/preprocessed/X_test.csv')
y_train = np.ravel(pd.read_csv('data/preprocessed/y_train.csv'))
y_test = np.ravel(pd.read_csv('data/preprocessed/y_test.csv'))

# Train model
rf_classifier = ensemble.RandomForestClassifier(n_jobs=-1)
rf_classifier.fit(X_train, y_train)

# Test model
accuracy = rf_classifier.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

test_data = X_test.iloc[[0]]
test_data.to_csv('data/test_data.txt', sep=',', index=False)
print(f"Actual label: {y_test[0]}")
print(f"Predicted label: {rf_classifier.predict(test_data)[0]}")

# Save model
model_ref = bentoml.sklearn.save_model("accidents_rf", rf_classifier)
print(f"Model saved as: {model_ref}")