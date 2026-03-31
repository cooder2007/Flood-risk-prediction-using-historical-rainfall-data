# 🌊 FloodSense AI

> AI-powered flood risk prediction system with interactive map and visualization.

---

## 🚀 Overview

FloodSense AI is a web-based application that predicts flood risk based on environmental parameters like rainfall, river level, and flood level.

It integrates:
- 🤖 Machine Learning models
- 🗺️ Interactive Map Visualization
- 📊 Dynamic Charts
- ⚡ Real-time Predictions via API

---

## ✨ Features

- 🎯 Predict flood risk (High / Medium / Low)
- 🧠 Multiple ML models:
  - KNN
  - Decision Tree
  - Logistic Regression
- 🗺️ Interactive map with colored markers
- 📊 Chart visualization of inputs
- ⚡ Fast API-based backend
- 🎨 Clean UI (card-based design)

---

## 🛠️ Tech Stack

### Frontend
- HTML, CSS, JavaScript
- Leaflet.js (Map)
- Chart.js (Graphs)

### Backend
- Python (Flask)
- Scikit-learn (ML models)

---

## 📂 Project Structure

FloodSense-AI/
│
├── app.py 
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── maps/
│       ├── india_district.geojson
│       └── india_states.geojson
│
└── README.md

## 📂 Codes:

### app.py :

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
    'dt': (dt, round(accuracy_score(y_test, dt.predict(X_test)) * 100, 1)),
    'lr': (lr, round(accuracy_score(y_test, lr.predict(X_test)) * 100, 1)),
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

    rain = float(data['rainfall_mm'])
    river = float(data['river_level_m'])
    flood = float(data['flood_level_m'])
    model_name = data.get('model', 'knn')

    X_input = scaler.transform([[rain, river, flood]])

    model, acc = MODELS.get(model_name, MODELS['knn'])

    pred = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]

    label = le.inverse_transform([pred])[0]

    classes = list(le.classes_)
    prob_dict = {classes[i]: round(float(p)*100, 1) for i, p in enumerate(probs)}

    score_map = {'Low': 25, 'Medium': 60, 'High': 90}

    return jsonify({
        "risk_level": label,
        "risk_score": score_map[label],
        "probabilities": prob_dict,
        "model_used": model_name.upper(),
        "model_accuracy": acc,
        "flood_likely": label != "Low"
    })

if __name__ == '__main__':
    app.run(debug=True)

### templates/index.html:

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FloodSense AI</title>
  <link rel="stylesheet" href="/static/style.css">
  <script src="https://d3js.org/d3.v7.min.js"></script>
</head>

<body>

<h1>🌊 FloodSense AI</h1>

<div class="container">

  <div class="map-card">
    <h2>India Flood Risk Map</h2>
    <div id="map"></div>
  </div>

  <div class="card">

    <label>Rainfall (mm)</label>
    <input type="number" id="rain" value="100">

    <label>River Level (m)</label>
    <input type="number" id="river" value="1.5">

    <label>Flood Level (m)</label>
    <input type="number" id="flood" value="1.2">

    <!-- ✅ FIX 1: Separate, correctly-labelled selects for State and Model -->
    <label>State</label>
    <select id="state">
      <option value="UP">Uttar Pradesh</option>
      <option value="MH">Maharashtra</option>
      <option value="DL">Delhi</option>
    </select>

    <label>Model</label>
    <select id="model">
      <option value="knn">KNN</option>
      <option value="dt">Decision Tree</option>
      <option value="lr">Logistic Regression</option>
    </select>

    <button onclick="predict()">⚡ Predict</button>

    <!-- ✅ FIX 2: Only ONE result div, inside the card -->
    <div id="result"></div>

  </div>

</div>

<!-- ❌ REMOVED: duplicate <div id="result"> that was here -->

<script>
const STATE_URL = "/static/maps/india_states.geojson";

const COLORS = {
  Low: "#22c55e",
  Medium: "#f59e0b",
  High: "#ef4444"
};

function normalizeState(name) {
  // ✅ FIX 3: guard against undefined before lookup
  if (!name) return "Unknown";
  const map = {
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "NCT of Delhi": "Delhi",
    "Orissa": "Odisha"
  };
  return map[name] || name;
}

async function loadMap() {
  const width = 500;
  const height = 600;

  const svg = d3.select("#map")
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  const geo = await fetch(STATE_URL).then(r => r.json());

  const projection = d3.geoMercator().fitSize([width, height], geo);
  const path = d3.geoPath().projection(projection);

  svg.selectAll("path")
    .data(geo.features)
    .enter()
    .append("path")
    .attr("d", path)
    .attr("fill", "#0f172a")
    .attr("stroke", "#00c8ff")
    .attr("stroke-width", 0.5)
    .on("mouseover", function () {
      d3.select(this).attr("fill", "#1e293b");
    })
    .on("mouseout", function () {
      d3.select(this).attr("fill", "#0f172a");
    })
    .on("click", function (e, d) {
      // ✅ FIX 3 continued: try all known property keys
      const rawName =
        d.properties.st_nm ||
        d.properties.NAME_1 ||
        d.properties.name ||
        d.properties.State ||
        d.properties.state;

      const state = normalizeState(rawName);

      document.getElementById("result").innerHTML =
        `<p>📍 Selected: <b>${state}</b></p>`;
    });
}

loadMap();

async function predict() {
  const state = document.getElementById('state').value;
  const rain  = document.getElementById('rain').value;
  const river = document.getElementById('river').value;
  const flood = document.getElementById('flood').value;
  const model = document.getElementById('model').value; // ✅ now works

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      rainfall_mm:   rain,
      river_level_m: river,
      flood_level_m: flood,
      model:         model,
      state:         state
    })
  });

  const data = await res.json();

  document.getElementById('result').innerHTML = `
    <h2>Result</h2>
    <p><b>Risk:</b> <span style="color:${COLORS[data.risk_level]}">${data.risk_level}</span></p>
    <p><b>Score:</b> ${data.risk_score}</p>
    <p><b>Model:</b> ${data.model_used} (${data.model_accuracy}%)</p>
  `;
}
</script>

</body>
</html>

### static/style.css:

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #0a192f;
  color: #e2e8f0;
  text-align: center;
  min-height: 100vh;
}

h1 {
  padding: 24px 0 16px;
  font-size: 2rem;
  color: #00c8ff;
  letter-spacing: 1px;
}

.container {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: flex-start;
  gap: 24px;
  padding: 0 24px 40px;
  max-width: 1300px;
  margin: 0 auto;
}

.map-card {
  background: #112240;
  border-radius: 12px;
  padding: 20px;
  flex: 0 0 auto;
}

.map-card h2 {
  margin-bottom: 12px;
  color: #94a3b8;
  font-size: 1rem;
  text-transform: uppercase;
  letter-spacing: 1px;
}

#map svg { display: block; }


.right-panel {
  display: flex;
  flex-direction: column;
  gap: 20px;
  flex: 1 1 320px;
  max-width: 680px;
}


.card {
  background: #112240;
  padding: 24px;
  border-radius: 12px;
  text-align: left;
}

.card label {
  display: block;
  font-size: 0.8rem;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 12px 0 4px;
}

input, select {
  width: 100%;
  padding: 9px 12px;
  background: #0f172a;
  border: 1px solid #1e3a5f;
  border-radius: 6px;
  color: #e2e8f0;
  font-size: 0.95rem;
}

input:focus, select:focus {
  outline: none;
  border-color: #00c8ff;
}

button {
  width: 100%;
  margin-top: 18px;
  padding: 12px;
  background: linear-gradient(135deg, #00c8ff, #0070f3);
  border: none;
  border-radius: 8px;
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

button:hover { opacity: 0.88; }


#result { margin-top: 16px; }

.result-box {
  background: #0f172a;
  border-radius: 8px;
  padding: 14px 16px;
  text-align: left;
  line-height: 1.9;
}

.result-box p { font-size: 0.95rem; }

.badge {
  display: inline-block;
  padding: 2px 8px;
  background: #1e3a5f;
  border-radius: 99px;
  font-size: 0.75rem;
  color: #00c8ff;
}

.selected-state { padding: 8px 0; color: #94a3b8; }
.loading { color: #94a3b8; font-style: italic; }


.charts-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.chart-card {
  flex: 1 1 260px;
  background: #112240;
  border-radius: 12px;
  padding: 20px;
}

.chart-card h3 {
  font-size: 0.85rem;
  color: #94a3b8;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 14px;
}

### static/maps/india_district.geojson:

--Unattachable due to very high size:

### static/maps/india_states.geojson

--Unattachable due to very high size:
