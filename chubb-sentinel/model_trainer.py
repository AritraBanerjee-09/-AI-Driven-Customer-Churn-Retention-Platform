# model_trainer.py
import xgboost as xgb
import shap
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

def train_model():
    with tqdm(total=6, desc="Overall Training Progress") as pbar:
        
        # --- Step 1: Load Data ---
        pbar.set_description("Step 1/6: Loading data...")
        X_train = pd.read_csv('data/X_train.csv')
        X_test = pd.read_csv('data/X_test.csv')
        y_train = pd.read_csv('data/y_train.csv').squeeze()
        y_test = pd.read_csv('data/y_test.csv').squeeze()
        preprocessor = joblib.load('artifacts/preprocessor.joblib')
        pbar.update(1)

        # --- Step 2: Transform Data ---
        pbar.set_description("Step 2/6: Transforming data...")
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        pbar.update(1)

        # --- Step 3: Apply SMOTE ---
        pbar.set_description("Step 3/6: Applying SMOTE (long step)...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
        pbar.update(1)

        # --- Step 4: Train XGBoost Model ---
        pbar.set_description("Step 4/6: Training XGBoost model...")
        model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
        model.fit(X_train_smote, y_train_smote)
        pbar.update(1)
        
        # --- Step 5: Save Model Artifact ---
        pbar.set_description("Step 5/6: Saving model...")
        joblib.dump(model, 'artifacts/xgb_model.joblib')
        pbar.update(1)

        # --- Step 6: Create and Save SHAP Explainer ---
        pbar.set_description("Step 6/6: Creating SHAP explainer...")
        feature_names = preprocessor.get_feature_names_out()
        X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
        explainer = shap.TreeExplainer(model, X_train_processed_df)
        joblib.dump(explainer, 'artifacts/shap_explainer.joblib')
        pbar.update(1)

    # --- Evaluation ---
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_processed)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("------------------------\n")
    print("🎉 Model training process complete. Artifacts are saved.")

if __name__ == '__main__':
    train_model()