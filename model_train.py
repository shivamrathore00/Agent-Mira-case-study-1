import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
file_name = 'Case Study 1 Data.xlsx' # Ensure this matches your filename
df = pd.read_excel(file_name)

# 2. Feature Engineering
df['Date Sold'] = pd.to_datetime(df['Date Sold'])
df['Property_Age'] = df['Date Sold'].dt.year - df['Year Built']

# Drop rows where the target (Price) is missing, as we can't train on unknown answers
df = df.dropna(subset=['Price'])

# 3. Define Features and Target
features = ['Location', 'Size', 'Bedrooms', 'Bathrooms', 'Condition', 'Type', 'Property_Age']
target = 'Price'

X = df[features]
y = df[target]

# 4. Preprocessing Pipeline with Imputation (Handling NaNs)
numeric_features = ['Size', 'Bedrooms', 'Bathrooms', 'Property_Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fills missing numbers with the middle value
    ('scaler', StandardScaler())
])

categorical_features = ['Location', 'Condition', 'Type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing text with the most common item
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Define Model (Multi-core)
# n_jobs=-1 fulfills the multi-core processing requirement [cite: 21]
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Create the final Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# 6. Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 7. Evaluate
predictions = pipeline.predict(X_test)
print(f"Model Accuracy (R2 Score): {r2_score(y_test, predictions):.2f}")
print(f"Average Error: ${mean_absolute_error(y_test, predictions):,.2f}")

# 8. Save the Model [cite: 20]
joblib.dump(pipeline, 'house_price_model.joblib')

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load Data
file_name = 'Case Study 1 Data.xlsx' # Ensure this matches your filename
df = pd.read_excel(file_name)

# 2. Feature Engineering
df['Date Sold'] = pd.to_datetime(df['Date Sold'])
df['Property_Age'] = df['Date Sold'].dt.year - df['Year Built']

# Drop rows where the target (Price) is missing, as we can't train on unknown answers
df = df.dropna(subset=['Price'])

# 3. Define Features and Target
features = ['Location', 'Size', 'Bedrooms', 'Bathrooms', 'Condition', 'Type', 'Property_Age']
target = 'Price'

X = df[features]
y = df[target]

# 4. Preprocessing Pipeline with Imputation (Handling NaNs)
numeric_features = ['Size', 'Bedrooms', 'Bathrooms', 'Property_Age']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fills missing numbers with the middle value
    ('scaler', StandardScaler())
])

categorical_features = ['Location', 'Condition', 'Type']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Fills missing text with the most common item
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. Define Model (Multi-core)
# n_jobs=-1 fulfills the multi-core processing requirement [cite: 21]
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Create the final Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# 6. Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 7. Evaluate
predictions = pipeline.predict(X_test)
print(f"Model Accuracy (R2 Score): {r2_score(y_test, predictions):.2f}")
print(f"Average Error: ${mean_absolute_error(y_test, predictions):,.2f}")

# 8. Save the Model [cite: 20]
joblib.dump(pipeline, 'house_price_model.joblib')

print("\nModel saved successfully as 'house_price_model.joblib'")