from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'flood_data.csv')
df = pd.read_csv(DATA_FILE)

features = ['rainfall_mm', 'river_level_m', 'flood_level_m']
X = df[features].values
y = df['risk_level'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=4)
dt.fit(X_train, y_train)

lr = LogisticRegression(max_iter=500)
lr.fit(X_train, y_train)

MODELS = {
    'knn': (knn, round(accuracy_score(y_test, knn.predict(X_test)) * 100, 1)),
    'dt':  (dt,  round(accuracy_score(y_test, dt.predict(X_test))  * 100, 1)),
    'lr':  (lr,  round(accuracy_score(y_test, lr.predict(X_test))  * 100, 1)),
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/model_stats')
def stats():
    return jsonify({k: v[1] for k, v in MODELS.items()})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()

    rain  = float(data['rainfall_mm'])
    river = float(data['river_level_m'])
    flood = float(data['flood_level_m'])
    model_name = data.get('model', 'knn')

    X_input = scaler.transform([[rain, river, flood]])
    model, acc = MODELS.get(model_name, MODELS['knn'])

    pred  = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    label = le.inverse_transform([pred])[0]

    classes  = list(le.classes_)
    prob_dict = {classes[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}
    score_map = {'Low': 25, 'Medium': 60, 'High': 90}

    return jsonify({
        "risk_level":     label,
        "risk_score":     score_map[label],
        "probabilities":  prob_dict,
        "model_used":     model_name.upper(),
        "model_accuracy": acc,
        "flood_likely":   label != "Low"
    })

if __name__ == '__main__':
    app.run(debug=True, port=3000)   # ✅ FIX: match the port your browser uses