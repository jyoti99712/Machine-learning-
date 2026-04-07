# Import required libraries
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load real-world dataset
data = load_breast_cancer()

# Convert into DataFrame (optional, for viewing)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Features and target
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# List of kernels to test
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# Train and evaluate SVM with different kernels
for kernel in kernels:
    print("\n" + "="*50)
    print(f"SVM with {kernel} kernel")
    print("="*50)

    # Create SVM model
    svm_model = SVC(kernel=kernel)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = svm_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))