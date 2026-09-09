#---------------------------------------------------------------
# MLOps Assignment 1 - House Price Prediction
# Student ID: l232598
# Minahil Azeem
#---------------------------------------------------------------
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#---------------------------------------------------------------
# # 1. CONFIGURATION
STUDENT_ID = "l232598"

DATA_PATH = "data/dataset.csv"
MODEL_PATH = f"model/house_price_model_{STUDENT_ID}.pkl"

TARGET_COLUMN = "Price"
TEST_SIZE = 0.20
RANDOM_STATE = 42
LEARNING_RATE = 0.01

#---------------------------------------------------------------
# # 2. LOAD DATASET
print("=" * 60)
print("HOUSE PRICE PREDICTION")
print(f"Student ID: {STUDENT_ID}")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Number of rows    : {df.shape[0]}")
print(f"Number of columns : {df.shape[1]}")


#---------------------------------------------------------------
# # 3. DATA INSPECTION
print("\n[2] Inspecting dataset...")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

#---------------------------------------------------------------
# # 4. BASIC DATA CLEANING

print("\n[3] Performing basic data cleaning...")

# Remove duplicate rows
duplicates = df.duplicated().sum()

if duplicates > 0:
    print(f"Removing {duplicates} duplicate rows...")
    df = df.drop_duplicates()
else:
    print("No duplicate rows found.")

# Make sure target column exists
if TARGET_COLUMN not in df.columns:
    raise ValueError(
        f"Target column '{TARGET_COLUMN}' was not found in the dataset."
    )

# Remove rows where target value is missing
missing_target = df[TARGET_COLUMN].isnull().sum()

if missing_target > 0:
    print(f"Removing {missing_target} rows with missing Price...")
    df = df.dropna(subset=[TARGET_COLUMN])

print(f"Dataset shape after cleaning: {df.shape}")


#---------------------------------------------------------------
# # 5. REMOVE UNNECESSARY FEATURES

print("\n[4] Removing unnecessary features...")

# Id is an identifier, not a meaningful predictive feature
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
    print("Removed 'Id' column.")

else:
    print("'Id' column not found. Nothing to remove.")


#---------------------------------------------------------------
# # 6. SEPARATE FEATURES AND TARGET

print("\n[5] Separating features and target...")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print("Target variable:", TARGET_COLUMN)
print("Number of features:", X.shape[1])

print("\nFeatures:")
for feature in X.columns:
    print(f"  - {feature}")

#---------------------------------------------------------------
#  7. IDENTIFY FEATURE TYPES

print("\n[6] Identifying numerical and categorical features...")

numerical_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


#---------------------------------------------------------------
# # 8. NUMERICAL PREPROCESSING

print("\n[7] Creating numerical preprocessing pipeline...")

numerical_pipeline = Pipeline(
    steps=[
        # Replace missing numerical values with median
        ("imputer", SimpleImputer(strategy="median")),

        # Normalize numerical features
        ("scaler", StandardScaler())
    ]
)


#---------------------------------------------------------------
# # 9. CATEGORICAL PREPROCESSING

print("[8] Creating categorical preprocessing pipeline...")

categorical_pipeline = Pipeline(
    steps=[
        # Replace missing categorical values with most common value
        ("imputer", SimpleImputer(strategy="most_frequent")),

        # Convert categorical values into numerical representation
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

#---------------------------------------------------------------
# 10. COMBINE PREPROCESSING

print("[9] Combining preprocessing pipelines...")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


#---------------------------------------------------------------
# 11. DEFINE MACHINE LEARNING MODEL

print("\n[10] Creating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


#---------------------------------------------------------------
# 12. CREATE COMPLETE ML PIPELINE

print("[11] Creating complete ML pipeline...")

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)

#---------------------------------------------------------------
# 13. TRAIN-TEST SPLIT

print("\n[12] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

#---------------------------------------------------------------
# 14. TRAIN MODEL

print("\n[13] Training model...")
print("Please wait...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


#---------------------------------------------------------------
# 15. MAKE PREDICTIONS

print("\n[14] Making predictions...")

y_pred = pipeline.predict(X_test)

print("Predictions generated successfully!")

#---------------------------------------------------------------
# 16. MODEL EVALUATION

print("\n[15] Evaluating model...")

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5


print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

print("=" * 60)


#---------------------------------------------------------------
# 17. SAVE TRAINED MODEL

print("\n[16] Saving trained model...")

# Make sure model directory exists
os.makedirs("model", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(f"Model saved successfully!")
print(f"Model path: {MODEL_PATH}")

#---------------------------------------------------------------
# 18. FINAL MESSAGE

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Student ID : {STUDENT_ID}")
print(f"Model      : {MODEL_PATH}")
print(f"R² Score   : {r2:.4f}")
print("=" * 60)