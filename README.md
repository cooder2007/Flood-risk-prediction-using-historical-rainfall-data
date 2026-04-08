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
├── flood_data.csv
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

# FIX 1: Use abspath so the path works regardless of where you run python from
DATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'flood_data.csv')

# FIX 2: Validate CSV columns before training — gives clear error instead of cryptic crash
required_cols = {'rainfall_mm', 'river_level_m', 'flood_level_m', 'risk_level'}
df = pd.read_csv(DATA_FILE)
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"flood_data.csv is missing required columns: {missing}")

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
    # FIX 3: Wrap entire predict in try/except so server never crashes on bad input
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        rain  = float(data['rainfall_mm'])
        river = float(data['river_level_m'])
        flood = float(data['flood_level_m'])
        model_name = data.get('model', 'knn')

        X_input = scaler.transform([[rain, river, flood]])
        model, acc = MODELS.get(model_name, MODELS['knn'])

        pred  = model.predict(X_input)[0]
        probs = model.predict_proba(X_input)[0]
        label = le.inverse_transform([pred])[0]

        classes   = list(le.classes_)
        prob_dict = {classes[i]: round(float(p) * 100, 1) for i, p in enumerate(probs)}
        score_map = {'Low': 25, 'Medium': 60, 'High': 90}

        return jsonify({
            "risk_level":     label,
            "risk_score":     score_map.get(label, 50),  # FIX 4: .get() avoids KeyError crash
            "probabilities":  prob_dict,
            "model_used":     model_name.upper(),
            "model_accuracy": acc,
            "flood_likely":   label != "Low"
        })

    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3000)

### templates/index.html:

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FloodSense AI</title>
  <link rel="stylesheet" href="/static/style.css"><!-- FIX 1: was ../static/style.css -->
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

<h1>🌊 FloodSense AI</h1>

<div class="container">

  <div class="map-card">
    <h2>India Flood Risk Map</h2>
    <div id="map"></div>
  </div>

  <div class="right-panel">
    <div class="card">
      <label>Rainfall (mm)</label>
      <input type="number" id="rain" value="100">

      <label>River Level (m)</label>
      <input type="number" id="river" value="1.5">

      <label>Flood Level (m)</label>
      <input type="number" id="flood" value="1.2">

      <label>State</label>
      <!-- FIX 2: All 32 states from flood_data.csv (was only 3) -->
      <select id="state">
        <option value="Andaman & Nicobar Islands">Andaman &amp; Nicobar Islands</option>
        <option value="Andhra Pradesh">Andhra Pradesh</option>
        <option value="Arunachal Pradesh">Arunachal Pradesh</option>
        <option value="Assam">Assam</option>
        <option value="Bihar">Bihar</option>
        <option value="Chhattisgarh">Chhattisgarh</option>
        <option value="Delhi">Delhi</option>
        <option value="Goa">Goa</option>
        <option value="Gujarat">Gujarat</option>
        <option value="Haryana">Haryana</option>
        <option value="Himachal Pradesh">Himachal Pradesh</option>
        <option value="Jammu & Kashmir">Jammu &amp; Kashmir</option>
        <option value="Jharkhand">Jharkhand</option>
        <option value="Karnataka">Karnataka</option>
        <option value="Kerala">Kerala</option>
        <option value="Lakshadweep">Lakshadweep</option>
        <option value="Madhya Pradesh">Madhya Pradesh</option>
        <option value="Maharashtra">Maharashtra</option>
        <option value="Manipur">Manipur</option>
        <option value="Meghalaya">Meghalaya</option>
        <option value="Mizoram">Mizoram</option>
        <option value="Nagaland">Nagaland</option>
        <option value="Odisha">Odisha</option>
        <option value="Punjab">Punjab</option>
        <option value="Rajasthan">Rajasthan</option>
        <option value="Sikkim">Sikkim</option>
        <option value="Tamil Nadu">Tamil Nadu</option>
        <option value="Telangana">Telangana</option>
        <option value="Tripura">Tripura</option>
        <option value="Uttar Pradesh" selected>Uttar Pradesh</option>
        <option value="Uttarakhand">Uttarakhand</option>
        <option value="West Bengal">West Bengal</option>
      </select>

      <label>Model</label>
      <select id="model">
        <option value="knn">KNN</option>
        <option value="dt">Decision Tree</option>
        <option value="lr">Logistic Regression</option>
      </select>

      <button onclick="predict()">⚡ Predict</button>
      <div id="result"></div>
    </div>

    <div class="charts-row" id="charts-section" style="display:none;">
      <div class="chart-card">
        <h3>📊 Risk Probability — Bar</h3>
        <canvas id="barChart"></canvas>
      </div>
      <div class="chart-card">
        <h3>🥧 Risk Probability — Pie</h3>
        <canvas id="pieChart"></canvas>
      </div>
    </div>
  </div>

</div>

<script>
const STATE_URL = "/static/maps/india_states.geojson";
const COLORS = { Low: "#22c55e", Medium: "#f59e0b", High: "#ef4444" };

let barChartInstance = null;
let pieChartInstance = null;

function normalizeState(name) {
  if (!name) return "Unknown";
  const map = {
    "Jammu and Kashmir":           "Jammu & Kashmir",
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "NCT of Delhi":                "Delhi",
    "Orissa":                      "Odisha"
  };
  return map[name] || name;
}

async function loadMap() {
  try {
    const width = 480, height = 560;

    const svg = d3.select("#map")
      .append("svg")
      .attr("width", width)
      .attr("height", height);

    const res = await fetch(STATE_URL);
    if (!res.ok) throw new Error(`GeoJSON fetch failed: ${res.status}`);
    const geo = await res.json();

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
      .on("mouseover", function () { d3.select(this).attr("fill", "#1e293b"); })
      .on("mouseout",  function () { d3.select(this).attr("fill", "#0f172a"); })
      .on("click", function (e, d) {
        const rawName =
          d.properties.st_nm  ||
          d.properties.NAME_1 ||
          d.properties.name   ||
          d.properties.State  ||
          d.properties.state;

        const stateName = normalizeState(rawName);

        // FIX 3: Map click now syncs the state dropdown
        const stateSelect = document.getElementById("state");
        let matched = false;
        for (let opt of stateSelect.options) {
          if (opt.value === stateName || opt.text === stateName) {
            stateSelect.value = opt.value;
            matched = true;
            break;
          }
        }

        document.getElementById("result").innerHTML =
          `<p class="selected-state">📍 Selected: <b>${stateName}</b>${matched ? '' : ' (not in dropdown)'}</p>`;
      });

  } catch (err) {
    console.error("Map error:", err);
    document.getElementById("map").innerHTML =
      `<p style="color:#ef4444;padding:20px">⚠️ Map failed: ${err.message}</p>`;
  }
}

loadMap();

function renderCharts(probs) {
  // FIX 4: Sort labels Low → Medium → High so colors always match correctly
  const ORDER = ['Low', 'Medium', 'High'];
  const entries = ORDER.filter(k => k in probs).map(k => [k, probs[k]]);
  const labels = entries.map(e => e[0]);
  const values = entries.map(e => e[1]);

  if (barChartInstance) barChartInstance.destroy();
  barChartInstance = new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Probability (%)',
        data: values,
        backgroundColor: labels.map(l => COLORS[l] || '#00c8ff'),
        borderRadius: 6,
        borderSkipped: false
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          beginAtZero: true, max: 100,
          ticks: { color: '#cbd5e1', callback: v => v + '%' },
          grid:  { color: '#1e293b' }
        },
        x: { ticks: { color: '#cbd5e1' }, grid: { display: false } }
      }
    }
  });

  if (pieChartInstance) pieChartInstance.destroy();
  pieChartInstance = new Chart(document.getElementById('pieChart'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: labels.map(l => COLORS[l] || '#00c8ff'),
        borderColor: '#0a192f',
        borderWidth: 3
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#cbd5e1', padding: 12 }
        },
        tooltip: {
          callbacks: { label: ctx => ` ${ctx.label}: ${ctx.raw}%` }
        }
      }
    }
  });
}

async function predict() {
  const state = document.getElementById('state').value;
  const rain  = document.getElementById('rain').value;
  const river = document.getElementById('river').value;
  const flood = document.getElementById('flood').value;
  const model = document.getElementById('model').value;

  document.getElementById('result').innerHTML =
    '<p class="loading">⏳ Predicting…</p>';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        rainfall_mm: rain, river_level_m: river,
        flood_level_m: flood, model, state
      })
    });

    const ct = res.headers.get('content-type') || '';
    if (!res.ok || !ct.includes('application/json')) {
      const text = await res.text();
      throw new Error(`Server error ${res.status}: ${text.slice(0, 120)}`);
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    const color = COLORS[data.risk_level] || '#fff';

    document.getElementById('result').innerHTML = `
      <div class="result-box" style="border-left: 4px solid ${color}">
        <p><b>Risk Level:</b>
          <span style="color:${color}; font-size:1.1em">${data.risk_level}</span>
        </p>
        <p><b>Risk Score:</b> ${data.risk_score} / 100</p>
        <p><b>Model:</b> ${data.model_used} &nbsp;
          <span class="badge">${data.model_accuracy}% acc</span>
        </p>
        <p><b>Flood Likely:</b>
          ${data.flood_likely
            ? '<span style="color:#ef4444">⚠️ Yes</span>'
            : '<span style="color:#22c55e">✅ No</span>'}
        </p>
      </div>
    `;

    document.getElementById('charts-section').style.display = 'flex';
    renderCharts(data.probabilities);

  } catch (err) {
    document.getElementById('result').innerHTML =
      `<p style="color:#ef4444">❌ Error: ${err.message}</p>`;
    console.error(err);
  }
}
</script>

</body>
</html>

### static/maps/india_district.geojson:

--Unattachable due to very high size:

### static/maps/india_states.geojson

--Unattachable due to very high size:
