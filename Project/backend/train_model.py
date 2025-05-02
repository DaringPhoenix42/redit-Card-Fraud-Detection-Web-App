import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """Load and preprocess the data."""
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    
    # Separate features and target
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Identify categorical columns
    categorical_cols = ['business_category', 'country', 'suspicious_activity_flag']
    
    # Create and fit label encoders for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        logger.info(f"Categories for {col}: {le.classes_}")
    
    # Scale numerical features
    numerical_cols = ['business_age', 'monthly_revenue', 'avg_transaction_value', 
                     'chargeback_ratio', 'negative_reviews_ratio', 'previous_fraud_reports']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    return X, y, label_encoders, scaler

def train_model(X, y):
    """Train the XGBoost model with hyperparameter tuning."""
    logger.info("Training model...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define parameter grid for XGBoost
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create base model
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    logger.info("\nBest parameters:")
    logger.info(grid_search.best_params_)
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    logger.info(f"\nROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    return best_model, X_test, y_test

def save_model_and_encoders(model, label_encoders, scaler):
    """Save the model and encoders."""
    logger.info("Saving model and encoders...")
    
    # Save model
    joblib.dump(model, 'Project/model/fraud_detection_model.joblib')
    
    # Save encoders
    joblib.dump(label_encoders, 'Project/model/label_encoders.joblib')
    
    # Save scaler
    joblib.dump(scaler, 'Project/model/scaler.joblib')
    
    logger.info("Model and encoders saved successfully!")

def main():
    # Load and preprocess data
    X, y, label_encoders, scaler = load_and_preprocess_data('Project/sample_fraud_data.csv')
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Save model and encoders
    save_model_and_encoders(model, label_encoders, scaler)

if __name__ == "__main__":
    main() 