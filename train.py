import os
import logging
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data():
    """Load CSV, handle missing values, and create target variables."""
    df = pd.read_csv(config.DATA_PATH)

    # Drop rows missing essential target columns
    df.dropna(subset=['flood_level', 'flood_occurred'], inplace=True)

    # Fill missing numeric features with median
    for col in ['rainfall_mm', 'river_level_m']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Create flood_risk based on flood_level bins
    bins = [-np.inf, 1.0, 2.0, np.inf]
    labels = [0, 1, 2]  # Low, Medium, High
    df['flood_risk'] = pd.cut(df['flood_level_m'], bins=bins, labels=labels).astype(int)

    X = df[['rainfall_mm', 'river_level_m']].values
    y_level = df['flood_level_m'].values
    y_risk = df['flood_risk'].values
    y_occurred = df['flood_occurred'].values

    return X, y_level, y_risk, y_occurred

def train_models(X, y_level, y_risk, y_occurred):
    """Split data, scale features, train three models, and return them along with scaler."""
    # Split
    X_train, X_test, y_level_train, y_level_test, y_risk_train, y_risk_test, y_occ_train, y_occ_test = train_test_split(
        X, y_level, y_risk, y_occurred, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_level_train)
    lr_pred = lr.predict(X_test_scaled)
    logging.info("Linear Regression RMSE: %.3f", np.sqrt(mean_squared_error(y_level_test, lr_pred)))

    # KNN Classifier (risk level)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_risk_train)
    knn_pred = knn.predict(X_test_scaled)
    logging.info("KNN Accuracy: %.3f", accuracy_score(y_risk_test, knn_pred))
    logging.info("KNN Classification Report:\n%s", classification_report(y_risk_test, knn_pred))

    # Logistic Regression (occurrence)
    log_reg = LogisticRegression(random_state=config.RANDOM_STATE)
    log_reg.fit(X_train_scaled, y_occ_train)
    log_reg_pred = log_reg.predict(X_test_scaled)
    logging.info("Logistic Regression Accuracy: %.3f", accuracy_score(y_occ_test, log_reg_pred))
    logging.info("Logistic Regression Classification Report:\n%s", classification_report(y_occ_test, log_reg_pred))

    return lr, knn, log_reg, scaler

def save_models(lr, knn, log_reg, scaler):
    """Save models and scaler to disk."""
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.MODEL_FILE, 'wb') as f:
        pickle.dump((lr, knn, log_reg, scaler), f)
    logging.info("Models saved to %s", config.MODEL_FILE)

if __name__ == '__main__':
    logging.info("Loading data...")
    X, y_level, y_risk, y_occurred = load_and_preprocess_data()
    logging.info("Training models...")
    lr, knn, log_reg, scaler = train_models(X, y_level, y_risk, y_occurred)
    save_models(lr, knn, log_reg, scaler)