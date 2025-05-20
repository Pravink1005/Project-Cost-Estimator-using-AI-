import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# 1. Load your data
df = pd.read_csv('data/project_cost_data_augmented.csv')

# 2. Clean & validate
df = df[df['Total_Cost'] > 1000]  # Remove bad rows if needed
X = df.drop(columns=['Total_Cost'])
y = df['Total_Cost']

# 3. Define preprocessing
cat_features = ['Project_Type']
num_features = [col for col in X.columns if col not in cat_features]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# 4. Define full pipeline (preprocessing + model)
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb)
])

# 5. Hyperparameter tuning (optional)
param_grid = {
    'model__n_estimators': [100],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
    'model__subsample': [0.7, 1.0]
}

# 6. Train/test split and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error')
search.fit(X_train, y_train)

# 7. Evaluate
y_pred = search.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Best Model RMSE: INR {rmse:.2f}")
print(f"Best Params: {search.best_params_}")

# 8. Save the full pipeline
joblib.dump(search.best_estimator_, 'models/best_xgb_pipeline.pkl')
print("Model pipeline saved as models/best_xgb_pipeline.pkl")
