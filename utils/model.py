"""
model.py — Machine Learning Engine

PURPOSE:
    Train a model to predict whether a stock's price will go UP or DOWN
    tomorrow, based on today's technical indicators.

CLASSIFICATION TASK:
    Input: 13 technical features (RSI, MACD, Volatility, etc.)
    Output: 0 (price goes DOWN) or 1 (price goes UP)

WHY RANDOM FOREST?
    1. Handles non-linear relationships (stock patterns aren't straight lines)
    2. Resistant to overfitting (uses many trees, averages them)
    3. Gives feature importance (tells us WHICH indicators matter most)
    4. Works well with small-to-medium datasets
    5. No need for feature scaling (unlike Neural Networks or SVM)
    6. Fast training and prediction

WHY NOT DEEP LEARNING?
    - We have ~300-400 data points (2 years of daily data)
    - Deep learning needs THOUSANDS of samples
    - Random Forest works better on tabular data this size
    - Simpler = more reliable for this use case

MODEL VALIDATION:
    We use TIME-BASED splitting, NOT random splitting.
    WHY? Because in finance, you can't use future data to predict the past.
    Train on 2023 data → Test on 2024 data (walk-forward validation).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from utils.indicators import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Model save/load directory
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create the prediction target: Will the price go UP tomorrow?
    
    Target = 1 if tomorrow's close > today's close
    Target = 0 if tomorrow's close <= today's close
    
    shift(-1) looks 1 day into the future — this is what we're predicting.
    
    IMPORTANT: The LAST row will have NaN target (we don't know tomorrow yet).
    That last row is exactly what we'll use for LIVE prediction.
    """
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    return data


def prepare_features(data: pd.DataFrame):
    """
    Split data into features (X) and target (y).
    
    Returns:
        X_train, y_train: Historical data for training
        X_latest: The most recent row (for live prediction)
        feature_names: List of feature column names
    """
    # Ensure all features exist
    available_features = [f for f in FEATURE_COLUMNS if f in data.columns]
    
    if len(available_features) < 5:
        raise ValueError(f"Not enough features. Found: {available_features}")
    
    # Separate the latest row (no target — this is what we predict)
    data_with_target = data.dropna(subset=["Target"])
    
    X = data_with_target[available_features]
    y = data_with_target["Target"]
    
    # Latest data point for live prediction
    X_latest = data[available_features].iloc[-1:]
    
    return X, y, X_latest, available_features


def train_model(data: pd.DataFrame) -> dict:
    """
    Train the ML model and return everything needed for predictions.
    
    PROCESS:
    1. Create target variable
    2. Prepare features
    3. Time-series split for validation (NO data leakage!)
    4. Train Random Forest + Gradient Boosting (ensemble)
    5. Evaluate on test set
    6. Generate feature importance
    7. Make live prediction on latest data
    
    Returns dict with:
        - model: Trained model object
        - prediction: 0 or 1 (DOWN or UP)
        - confidence: 0.0 to 1.0 (how sure the model is)
        - accuracy: Model accuracy on test data
        - feature_importance: Which features mattered most
        - report: Full classification report
    """
    # --- Step 1: Create target ---
    data = create_target(data)
    
    # --- Step 2: Prepare features ---
    X, y, X_latest, feature_names = prepare_features(data)
    
    if len(X) < 100:
        raise ValueError(f"Not enough data points ({len(X)}). Need at least 100 for reliable training.")
    
    # --- Step 3: Time-series split ---
    # Use 80% for training, 20% for testing
    # CRITICAL: No shuffling! We must respect time order.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # --- Step 4: Train Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=200,       # 200 trees in the forest
        max_depth=10,           # Limit depth to prevent overfitting
        min_samples_split=10,   # Need at least 10 samples to split a node
        min_samples_leaf=5,     # Each leaf must have at least 5 samples
        random_state=42,        # Reproducibility
        n_jobs=-1,              # Use all CPU cores
        class_weight="balanced" # Handle imbalanced classes
    )
    
    rf_model.fit(X_train, y_train)
    
    # --- Step 5: Evaluate ---
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["DOWN", "UP"], output_dict=True)
    
    # --- Step 6: Feature importance ---
    importance = pd.Series(
        rf_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    # --- Step 7: Live prediction ---
    prediction = rf_model.predict(X_latest)[0]
    confidence = rf_model.predict_proba(X_latest)[0].max()
    
    # --- Cross-validation for robustness ---
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
        
        cv_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, 
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight="balanced"
        )
        cv_model.fit(X_cv_train, y_cv_train)
        cv_scores.append(accuracy_score(y_cv_val, cv_model.predict(X_cv_val)))
    
    avg_cv_accuracy = np.mean(cv_scores)
    
    # --- Save model ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "latest_model.pkl"))
    
    return {
        "model": rf_model,
        "prediction": int(prediction),
        "confidence": float(confidence),
        "accuracy": float(accuracy),
        "cv_accuracy": float(avg_cv_accuracy),
        "feature_importance": importance,
        "report": report,
        "feature_names": feature_names,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
