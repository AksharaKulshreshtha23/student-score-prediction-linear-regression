# ==========================================
# Student Score Prediction using Linear Regression
# ==========================================

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("student_scores.csv")

print("First 5 rows:")
print(df.head())


# ==========================================
# 2. Define Features and Target
# ==========================================

X = df[["Hours"]]  # Feature
y = df["Score"]    # Target


# ==========================================
# 3. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 4. Model Training
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==========================================
# 5. Predictions
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 6. Model Evaluation
# ==========================================

print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# ==========================================
# 7. Visualization
# ==========================================

plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.title("Study Hours vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()
