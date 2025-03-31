import joblib
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # NEW

print(f"✅ Using scikit-learn version: {sklearn.__version__}")

# ✅ Load dataset
data = pd.read_csv("accidents_india.csv")
print("Dataset loaded successfully!\n", data.head())

# ✅ Convert categorical features to numeric
label_encoders = {}  # Dictionary to store label encoders

for column in data.select_dtypes(include=['object']).columns:  # Find categorical columns
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])  # Convert categories to numbers
    label_encoders[column] = le  # Save encoder for future use
    print(f"✅ Encoded '{column}'")  # Print confirmation

# ✅ Define features and target
X = data.iloc[:, :-1]  # Features (all except last column)
y = data.iloc[:, -1]   # Target (last column)

# ✅ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ✅ Save model and encoders
joblib.dump(model, 'test1.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')  # Save encoders for decoding predictions

print("✅ Model and encoders saved successfully!")
