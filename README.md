
This `README.md` contains all the code and instructions you need. Simply copy the content into a file named `README.md` in your project folder, and then create the three files (`app.py`, `templates/index.html`, `static/style.css`) with the code provided.
markdown
# Flood Prediction System

A machine learning‑based web application that predicts flood levels, flood risk, and flood occurrence using historical data. The system is built with Python (Flask) and uses three models:

- **Linear Regression** – predicts the flood level (in meters) based on rainfall and river level.
- **K‑Nearest Neighbors (KNN)** – classifies flood risk into **Low**, **Medium**, or **High**.
- **Logistic Regression** – performs binary classification (flood **Yes** or **No**).

The front‑end is a simple HTML form with CSS styling, allowing users to input new data and receive predictions in real time.

---

---

## File Structure
flood_prediction/
├── app.py               
├── train.py             
├── flood_data.csv       
├── templates/
│   └── index.html       
├── static/
│   └── style.css        
├── models/             
│   └── models.pkl
├── config.py        
├── requirements.txt
├── README.md
└── LICENSE

## Requirements

- Python 3.7 or higher
- The following Python packages: 
  - `flask`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install them with:

```

How to Run
Install dependencies:

bash
pip install -r requirements.txt
Train the models (run once):

bash
python train.py
Start the Flask app:

bash
python app.py
Open your browser at http://127.0.0.1:5000.

Dataset Format
The CSV file (flood_data.csv) should contain at least the following columns:

Column	Description
rainfall	Rainfall amount in millimeters (float)
river_level	River water level in meters (float)
flood_level	Observed flood level in meters (float) – used for regression
flood_occurred	Binary indicator (0 = no flood, 1 = flood)
Example rows:

text
rainfall,river_level,flood_level,flood_occurred
120.5,2.8,1.5,1
45.2,1.2,0.0,0
200.0,3.5,2.2,1
Missing values in rainfall and river_level are filled with the median. Rows with missing flood_level or flood_occurred are dropped.

How to Run
Place your dataset as flood_data.csv in the same folder as app.py.

Run the Flask application:

bash
python app.py
Open your browser and go to http://127.0.0.1:5000/.

Enter the rainfall (mm) and river level (m) and click Predict.

The system will display:

Predicted flood level (Linear Regression)

Flood risk category (KNN)

Whether a flood is predicted to occur (Logistic Regression)

Code
1. app.py – Flask Backend
python
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

# ------------------------------
# 1. Load and preprocess data
# ------------------------------
def load_and_preprocess_data(filepath='flood_data.csv'):
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Drop rows with missing target values
    df.dropna(subset=['flood_level', 'flood_occurred'], inplace=True)
    
    # For features, fill missing values with median
    for col in ['rainfall', 'river_level']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Create a flood_risk column for KNN (multi‑class)
    # Example thresholds: low (<1m), medium (1–2m), high (>2m)
    bins = [-np.inf, 1.0, 2.0, np.inf]
    labels = [0, 1, 2]   # 0=low, 1=medium, 2=high
    df['flood_risk'] = pd.cut(df['flood_level'], bins=bins, labels=labels).astype(int)
    
    # Features: rainfall and river_level
    X = df[['rainfall', 'river_level']].values
    y_level = df['flood_level'].values           # target for Linear Regression
    y_risk = df['flood_risk'].values             # target for KNN (multi‑class)
    y_occurred = df['flood_occurred'].values     # target for Logistic Regression
    
    return X, y_level, y_risk, y_occurred

# ------------------------------
# 2. Train models
# ------------------------------
def train_models(X, y_level, y_risk, y_occurred):
    # Split data (use same split for all models)
    X_train, X_test, y_level_train, y_level_test, y_risk_train, y_risk_test, y_occ_train, y_occ_test = train_test_split(
        X, y_level, y_risk, y_occurred, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2.1 Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_level_train)
    lr_pred = lr.predict(X_test_scaled)
    print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_level_test, lr_pred)))
    
    # 2.2 KNN Classifier (for flood risk)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_risk_train)
    knn_pred = knn.predict(X_test_scaled)
    print("KNN Accuracy:", accuracy_score(y_risk_test, knn_pred))
    print(classification_report(y_risk_test, knn_pred))
    
    # 2.3 Logistic Regression (binary flood occurrence)
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_scaled, y_occ_train)
    log_reg_pred = log_reg.predict(X_test_scaled)
    print("Logistic Regression Accuracy:", accuracy_score(y_occ_test, log_reg_pred))
    print(classification_report(y_occ_test, log_reg_pred))
    
    # Save models and scaler for later use (optional)
    with open('models.pkl', 'wb') as f:
        pickle.dump((lr, knn, log_reg, scaler), f)
    
    return lr, knn, log_reg, scaler

# ------------------------------
# 3. Flask web application
# ------------------------------
app = Flask(__name__)

# Load and train models once when the app starts
X, y_level, y_risk, y_occurred = load_and_preprocess_data()
lr_model, knn_model, log_reg_model, scaler = train_models(X, y_level, y_risk, y_occurred)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    try:
        rainfall = float(request.form['rainfall'])
        river_level = float(request.form['river_level'])
    except ValueError:
        return render_template('index.html', error="Please enter numeric values.")
    
    # Prepare input and scale
    input_data = np.array([[rainfall, river_level]])
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    flood_level_pred = lr_model.predict(input_scaled)[0]
    flood_risk_pred = knn_model.predict(input_scaled)[0]
    flood_occurrence_pred = log_reg_model.predict(input_scaled)[0]
    
    # Map numeric risk to text
    risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    flood_risk_text = risk_mapping.get(flood_risk_pred, 'Unknown')
    
    # Map occurrence to yes/no
    occurrence_text = 'Yes' if flood_occurrence_pred == 1 else 'No'
    
    # Render result
    return render_template('index.html',
                           rainfall=rainfall,
                           river_level=river_level,
                           flood_level=round(flood_level_pred, 2),
                           flood_risk=flood_risk_text,
                           flood_occurrence=occurrence_text)

if __name__ == '__main__':
    app.run(debug=True)
2. templates/index.html – HTML Form
html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Flood Prediction System</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <h1>Flood Prediction System</h1>
        <form action="/predict" method="post">
            <label for="rainfall">Rainfall (mm):</label>
            <input type="number" step="any" name="rainfall" required>

            <label for="river_level">River Level (m):</label>
            <input type="number" step="any" name="river_level" required>

            <button type="submit">Predict</button>
        </form>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if flood_level is not none %}
        <div class="result">
            <h3>Prediction Results</h3>
            <p><strong>Input:</strong> Rainfall = {{ rainfall }} mm, River Level = {{ river_level }} m</p>
            <p><strong>Predicted Flood Level:</strong> {{ flood_level }} m</p>
            <p><strong>Flood Risk (KNN):</strong> {{ flood_risk }}</p>
            <p><strong>Flood Occurrence (Logistic Regression):</strong> {{ flood_occurrence }}</p>
        </div>
        {% endif %}
    </div>
    <footer>
        Developed using Linear Regression, KNN, and Logistic Regression
    </footer>
</body>
</html>
3. static/style.css – External Stylesheet
css
/* static/style.css */
body {
    font-family: Arial, sans-serif;
    background-color: #f0f8ff;
    margin: 0;
    padding: 20px;
}

.container {
    max-width: 600px;
    margin: auto;
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h1 {
    text-align: center;
    color: #2c3e50;
}

label {
    display: block;
    margin-top: 10px;
    font-weight: bold;
}

input[type="number"] {
    width: 100%;
    padding: 8px;
    margin-top: 5px;
    border: 1px solid #ccc;
    border-radius: 4px;
}

button {
    background-color: #3498db;
    color: white;
    padding: 10px 15px;
    margin-top: 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    width: 100%;
}

button:hover {
    background-color: #2980b9;
}

.result {
    margin-top: 20px;
    padding: 15px;
    background-color: #ecf0f1;
    border-radius: 4px;
}

.error {
    color: red;
    margin-top: 10px;
}

footer {
    text-align: center;
    margin-top: 20px;
    font-size: 0.8em;
    color: #7f8c8d;
}
How It Works
Data Preprocessing: The CSV is read, missing values in features are filled with medians, rows with missing targets are dropped. A new column flood_risk (low/medium/high) is created based on the flood_level to serve as the target for KNN.

Model Training:

Linear Regression – predicts continuous flood level.

KNN – classifies flood risk into three categories.

Logistic Regression – predicts binary flood occurrence.
All features are scaled using StandardScaler before training.

Flask Routes:

GET / – displays the input form.

POST /predict – receives user input, scales it, and returns predictions from the three models.

Front‑end: The HTML form uses the external CSS for styling. Results are displayed dynamically using Jinja2 templates.

Customisation
Adjust the risk thresholds in load_and_preprocess_data() to match your local flood risk definitions.

Change the KNN parameter n_neighbors or any model hyperparameters to improve performance.

Save and load trained models using pickle to avoid retraining on every startup (the code already saves them as models.pkl; you can modify it to load if the file exists).

License
This project is open‑source and available under the MIT License.