import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Load dataset
try:
    df = pd.read_csv('d:/Report/data/project_cost_data_augmented.csv')
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'd:/Report/data/project_cost_data_augmented.csv'.")

# Validate dataset
if df.isnull().any().any():
    raise ValueError("Dataset contains missing values.")
if not np.issubdtype(df['Total_Cost'].dtype, np.number):
    raise ValueError("Total_Cost must be numeric.")

# Preprocess data
X = df.drop('Total_Cost', axis=1)
y = df['Total_Cost']

# Encode Project_Type
if 'Project_Type' in X.columns:
    X = pd.get_dummies(X, columns=['Project_Type'], drop_first=True, dtype=float)
else:
    print("Warning: 'Project_Type' column not found. Assuming already encoded.")

# Ensure numeric columns
X = X.apply(pd.to_numeric, errors='coerce')
if X.isnull().any().any():
    raise ValueError("Non-numeric data detected after conversion.")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

# Define model
xgb = XGBRegressor(random_state=42, device='cpu')
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Grid search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    error_score='raise'
)

try:
    grid_search.fit(X_train, y_train)
    print("Grid search completed successfully")
except Exception as e:
    print(f"Grid search failed: {str(e)}")
    raise

# Feature importance
best_model = grid_search.best_estimator_
feature_importance = pd.Series(best_model.feature_importances_, index=X_train.columns)
print("Feature Importance:")
print(feature_importance.sort_values(ascending=False))

# Save model and scaler
model_path = 'd:/Report/models/best_xgb_model.pkl'
scaler_path = 'd:/Report/models/scaler.pkl'
joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score (neg MSE): {grid_search.best_score_}")
print(f"Model saved to '{model_path}'")
print(f"Scaler saved to '{scaler_path}'")