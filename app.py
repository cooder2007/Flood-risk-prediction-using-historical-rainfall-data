import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import pickle
from flask import Flask, request, render_template



def load_and_preprocess_data(filepath='flood_data.csv'):
    
    df = pd.read_csv(filepath)
    
    
    df.dropna(subset=['flood_level', 'flood_occurred'], inplace=True)
    
    
    for col in ['rainfall', 'river_level']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    



    bins = [-np.inf, 1.0, 2.0, np.inf]
    labels = [0, 1, 2] 
    df['flood_risk'] = pd.cut(df['flood_level'], bins=bins, labels=labels).astype(int)
    

    X = df[['rainfall', 'river_level']].values
    y_level = df['flood_level'].values
    y_risk = df['flood_risk'].values
    y_occurred = df['flood_occurred'].values
    
    return X, y_level, y_risk, y_occurred


def train_models(X, y_level, y_risk, y_occurred):
    
    X_train, X_test, y_level_train, y_level_test, y_risk_train, y_risk_test, y_occ_train, y_occ_test = train_test_split(
        X, y_level, y_risk, y_occurred, test_size=0.2, random_state=42
    )
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_level_train)
    lr_pred = lr.predict(X_test_scaled)
    print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_level_test, lr_pred)))
    
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_risk_train)
    knn_pred = knn.predict(X_test_scaled)
    print("KNN Accuracy:", accuracy_score(y_risk_test, knn_pred))
    print(classification_report(y_risk_test, knn_pred))
    
    
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_occ_train)
    log_reg_pred = log_reg.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_occ_test, log_reg_pred))
    print(classification_report(y_occ_test, log_reg_pred))
    
    
    with open('models.pkl', 'wb') as f:
        pickle.dump((lr, knn, log_reg, scaler), f)
    
    return lr, knn, log_reg, scaler


app = Flask(__name__)


X, y_level, y_risk, y_occurred = load_and_preprocess_data()
lr_model, knn_model, log_reg_model, scaler = train_models(X, y_level, y_risk, y_occurred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        rainfall = float(request.form['rainfall'])
        river_level = float(request.form['river_level'])
    except ValueError:
        return render_template('index.html', error="Please enter numeric values.")
    
    
    input_data = np.array([[rainfall, river_level]])
    input_scaled = scaler.transform(input_data)
    
    
    flood_level_pred = lr_model.predict(input_scaled)[0]
    flood_risk_pred = knn_model.predict(input_scaled)[0]
    flood_occurrence_pred = log_reg_model.predict(input_scaled)[0]
    
    
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    flood_risk_text = risk_mapping.get(flood_risk_pred, 'Unknown')
    
    
    occurrence_text = 'Yes' if flood_occurrence_pred == 1 else 'No'
    
    
    return render_template('index.html',
                           rainfall=rainfall,
                           river_level=river_level,
                           flood_level=round(flood_level_pred, 2),
                           flood_risk=flood_risk_text,
                           flood_occurrence=occurrence_text)

if __name__ == '__main__':
    app.run(debug=True)