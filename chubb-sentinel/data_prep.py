# data_prep.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import joblib
from math import radians, sin, cos, sqrt, atan2

# --- Feature Engineering Functions ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dLat, dLon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def create_features(df):
    df['cust_orig_date'] = pd.to_datetime(df['cust_orig_date'], errors='coerce')
    df['orig_year'] = df['cust_orig_date'].dt.year
    df['orig_month'] = df['cust_orig_date'].dt.month
    household_counts = df['address_id'].value_counts().rename('household_size')
    df = df.merge(household_counts, left_on='address_id', right_index=True, how='left')
    dallas_center = (32.7767, -96.7970)
    fort_worth_center = (32.7555, -97.3308)
    df['distance_to_dallas'] = df.apply(lambda row: haversine(row['latitude'], row['longitude'], dallas_center[0], dallas_center[1]), axis=1)
    df['distance_to_fort_worth'] = df.apply(lambda row: haversine(row['latitude'], row['longitude'], fort_worth_center[0], fort_worth_center[1]), axis=1)
    return df

# --- Main Preprocessing Function ---
def preprocess_data(filepath='../archive/autoinsurance_churn.csv', sample_mode=True):
    # --- 1. Load Data ---
    if sample_mode:
        print("🚀 RUNNING IN SAMPLE MODE: Loading only the first 5,000 rows.")
        df = pd.read_csv(filepath, nrows=5000)
    else:
        print("🚀 RUNNING IN FULL MODE: Loading the entire dataset.")
        df = pd.read_csv(filepath)

    # --- 2. Clean and Engineer Features ---
    df['Churn'] = df['acct_suspd_date'].notna().astype(int)
    df = create_features(df)
    
    drop_features = ['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'latitude', 'longitude', 'acct_suspd_date', 'Churn']
    X = df.drop(columns=drop_features)
    y = df['Churn']
    
    # Impute missing values
    for col in X.select_dtypes(include=np.number).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].fillna(X[col].mode()[0])

    # --- 3. DYNAMICALLY Identify Feature Types ---
    ordinal_feature_name = 'home_market_value'
    market_value_order = ['1000 - 24999', '25000 - 49999', '50000 - 74999', '75000 - 99999', '100000 - 124999', '125000 - 149999', '150000 - 174999', '175000 - 199999', '200000 - 224999', '225000 - 249999', '250000 - 299999', '300000 - 349999', '350000 - 399999', '400000 - 499999', '500000+']
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if ordinal_feature_name in categorical_features:
        categorical_features.remove(ordinal_feature_name)
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # --- 4. Create Preprocessing Pipelines ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('ord', OrdinalEncoder(categories=[market_value_order], handle_unknown='use_encoded_value', unknown_value=-1), [ordinal_feature_name])
        ],
        remainder='passthrough'
    )
    
    # --- 5. Split Data and Verify Shape ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor.fit(X_train)

    # --- ‼️ IMPORTANT DEBUG STEP ‼️ ---
    X_train_transformed_debug = preprocessor.transform(X_train)
    print("\n" + "="*50)
    print(f"✅ DEBUG CHECK: Shape of transformed data is: {X_train_transformed_debug.shape}")
    print("This should have many columns (e.g., > 100), not 1.")
    print("="*50 + "\n")
    
    # --- 6. Save Artifacts ---
    joblib.dump(preprocessor, 'artifacts/preprocessor.joblib')
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("Data preparation complete. Preprocessor and data splits have been saved.")

if __name__ == '__main__':
    preprocess_data(sample_mode=True)