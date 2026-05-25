import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("diabetes.csv")

# Show First 5 Rows
print("\nFIRST 5 ROWS")
print(data.head())

# Dataset Information
print("\nDATASET SHAPE")
print(data.shape)

print("\nCOLUMN NAMES")
print(data.columns)

print("\nDATASET INFO")
print(data.info())

print("\nSTATISTICAL SUMMARY")
print(data.describe())

# Check Missing Values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Visualization 1 - Outcome Count
data['Outcome'].value_counts().plot(kind='bar')

plt.title("Diabetes Count")
plt.xlabel("Outcome")
plt.ylabel("Count")

plt.show()

# Visualization 2 - Glucose Distribution
data['Glucose'].hist()

plt.title("Glucose Distribution")
plt.xlabel("Glucose")
plt.ylabel("Frequency")

plt.show()

# Split Data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Model
model = LogisticRegression(max_iter=1000)

# Train Model
model.fit(X_train, y_train)

# Prediction
prediction = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, prediction)

print("\nMODEL ACCURACY")
print(accuracy)

# Confusion Matrix
print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, prediction))

# Test New Patient Data
sample = [[2,120,70,20,85,28.5,0.5,33]]

result = model.predict(sample)

print("\nNEW PATIENT RESULT")

if result[0] == 1:
    print("Patient Has Diabetes")
else:
    print("Patient Does Not Have Diabetes")