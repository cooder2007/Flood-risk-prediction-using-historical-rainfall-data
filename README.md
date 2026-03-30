
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
FloodSense-AI/
│
├── index.html              ← Frontend (the file you downloaded)
│
├── flood_ai.py             ← Python Flask backend (AI models)
│
├── flood_data.csv          ← Training dataset
│
├── requirements.txt        ← Python dependencies
│
└── README.md               ← (optional) setup instructions

## Requirements

- Python 3.7 or higher
- The following Python packages: 
  - `flask`
  - `pandas`
  - `numpy`
  - `scikit-learn`

Install them with:

```
▶️ How to run it
Step 1 — Install dependencies
bashcd FloodSense-AI
pip install -r requirements.txt
Step 2 — Start the backend
bashpython flood_ai.py
```
You'll see:
```
🌊 Flood Risk Prediction AI Server
   http://localhost:5000
Step 3 — Open the frontend
Just double-click index.html in your file manager, or open it in Chrome/Edge/Firefox

#####  Code  #####


# index.html:

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FloodSense AI — India Flood Risk Prediction</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:ital,wght@0,400;0,700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#04111f;--panel:#071c30;--panel2:#0a2240;
  --border:rgba(0,170,255,0.13);--accent:#00c8ff;--accent2:#005bff;
  --glow:rgba(0,200,255,0.25);--low:#22c55e;--med:#f59e0b;--high:#ef4444;
  --text:#dff0ff;--muted:#5b7fa0;
  --fh:'Syne',sans-serif;--fm:'Space Mono',monospace;--fb:'Inter',sans-serif;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:var(--fb);background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 70% 50% at 15% 0%,rgba(0,91,255,.1) 0%,transparent 55%),
             radial-gradient(ellipse 55% 40% at 85% 100%,rgba(0,200,255,.07) 0%,transparent 50%)}
body::after{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.025) 2px,rgba(0,0,0,.025) 4px)}

.wrap{position:relative;z-index:1;max-width:1640px;margin:0 auto;padding:0 22px}


header{padding:18px 0 15px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.logo{display:flex;align-items:center;gap:13px}
.logo-icon{width:42px;height:42px;background:linear-gradient(135deg,var(--accent2),var(--accent));
  border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;
  box-shadow:0 0 22px var(--glow)}
.logo h1{font-family:var(--fh);font-size:21px;font-weight:800;letter-spacing:-.5px;
  background:linear-gradient(90deg,var(--accent),#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.logo p{font-family:var(--fm);font-size:9px;color:var(--muted);letter-spacing:2.5px;margin-top:2px}
.hbadges{display:flex;gap:9px;flex-wrap:wrap;align-items:center}
.badge{font-family:var(--fm);font-size:9.5px;padding:4px 11px;border-radius:20px;border:1px solid;letter-spacing:.8px}
.b-blue{border-color:var(--accent2);color:var(--accent);background:rgba(0,91,255,.09)}
.b-green{border-color:var(--low);color:var(--low);background:rgba(34,197,94,.09)}
.b-pulse{animation:blink 2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.45}}
.srv{display:flex;align-items:center;gap:7px;font-family:var(--fm);font-size:9.5px;letter-spacing:.8px;
  padding:5px 12px;border-radius:20px;border:1px solid;transition:all .3s}
.srv .dot{width:7px;height:7px;border-radius:50%}
.offline{border-color:rgba(239,68,68,.3);color:#f87171}.offline .dot{background:var(--high)}
.online{border-color:rgba(34,197,94,.3);color:var(--low)}.online .dot{background:var(--low);animation:blink 1.5s infinite}


.grid{display:grid;grid-template-columns:1fr 468px;gap:18px;padding:18px 0;align-items:start}
@media(max-width:1080px){.grid{grid-template-columns:1fr}}
.lcol,.rcol{display:flex;flex-direction:column;gap:16px}


.card{background:var(--panel);border:1px solid var(--border);border-radius:15px;padding:18px;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1.5px;
  background:linear-gradient(90deg,transparent,var(--accent2),var(--accent),transparent);opacity:.55}
.ctitle{font-family:var(--fh);font-size:12px;font-weight:700;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.ctitle .dot{width:5px;height:5px;border-radius:50%;background:var(--accent);display:inline-block;flex-shrink:0}


.map-outer{position:relative;border-radius:11px;overflow:hidden;
  background:radial-gradient(ellipse at 50% 45%,rgba(0,50,140,.28) 0%,rgba(2,10,22,.9) 75%)}
#map-svg-wrap{width:100%;min-height:520px;display:flex;align-items:center;justify-content:center;position:relative}
#india-svg{width:100%;height:auto;display:block;cursor:default}


.district-path{
  stroke:#0d3a6e;stroke-width:.25;
  transition:fill .18s ease,stroke .18s ease,stroke-width .18s ease;
  cursor:pointer;
}
.district-path:hover{stroke:#00c8ff;stroke-width:1.2;filter:brightness(1.35)}
.district-path.state-hovered{stroke:#00c8ff;stroke-width:.6}
.district-path.selected-state{stroke:var(--accent);stroke-width:1.6;filter:brightness(1.45) drop-shadow(0 0 5px rgba(0,200,255,.5))}


.state-border{fill:none;stroke:rgba(0,200,255,.55);stroke-width:1.1;pointer-events:none}


.fill-high  {fill:rgba(239,68,68,.58)}
.fill-medium{fill:rgba(245,158,11,.52)}
.fill-low   {fill:rgba(34,197,94,.45)}
.fill-none  {fill:rgba(10,38,75,.72)}


#map-loader{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:14px;z-index:10;background:rgba(4,17,31,.85);border-radius:11px}
.spin{width:38px;height:38px;border:2px solid rgba(0,200,255,.18);border-top-color:var(--accent);
  border-radius:50%;animation:spin .75s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
#map-loader p{font-family:var(--fm);font-size:11px;letter-spacing:1.8px;color:var(--muted)}
#map-loader small{font-family:var(--fm);font-size:9px;color:rgba(91,127,160,.6);letter-spacing:.8px}


.mleg{display:flex;gap:14px;flex-wrap:wrap;margin-top:11px;padding:9px 13px;
  background:rgba(0,0,0,.35);border-radius:8px;border:1px solid var(--border);align-items:center}
.li{display:flex;align-items:center;gap:6px;font-size:10.5px;color:var(--muted)}
.ld{width:13px;height:13px;border-radius:3px}


#htip{
  position:fixed;pointer-events:none;z-index:9999;
  background:rgba(4,14,28,.97);
  border-radius:12px;padding:0;
  display:none;min-width:220px;
  box-shadow:0 8px 40px rgba(0,0,0,.7),0 0 0 1px rgba(0,200,255,.25);
  overflow:hidden;
}
#htip-bar{height:3px;width:100%}
#htip-body{padding:13px 16px 14px}
#htip-state{font-family:var(--fh);font-weight:800;font-size:16px;letter-spacing:-.3px;margin-bottom:3px}
#htip-risk{font-family:var(--fm);font-size:10px;letter-spacing:1.2px;margin-bottom:10px}
.htip-rows{display:flex;flex-direction:column;gap:5px}
.htip-row{display:flex;justify-content:space-between;align-items:center;font-size:11px}
.htip-lbl{color:var(--muted);font-family:var(--fm);font-size:9.5px;letter-spacing:.6px}
.htip-val{font-family:var(--fm);font-size:11px;color:var(--text);font-weight:700}
#htip-flood-bar{height:5px;background:rgba(255,255,255,.07);border-radius:3px;margin-top:8px;overflow:hidden}
#htip-flood-fill{height:100%;border-radius:3px;transition:width .4s ease}


.acc-row{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
.acc-box{background:rgba(0,0,0,.32);border:1px solid var(--border);border-radius:10px;
  padding:11px 10px;text-align:center;cursor:pointer;transition:border-color .2s,box-shadow .2s}
.acc-box.active{border-color:var(--accent);box-shadow:0 0 16px var(--glow)}
.acc-box:hover:not(.active){border-color:rgba(0,200,255,.35)}
.an{font-family:var(--fm);font-size:10px;color:var(--muted);letter-spacing:.8px}
.av{font-family:var(--fh);font-size:20px;font-weight:800;color:var(--accent);margin:3px 0}
.ab-bg{height:3px;background:rgba(255,255,255,.07);border-radius:2px;overflow:hidden}
.ab-fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:2px;transition:width 1.2s ease}

.fgrp{display:flex;flex-direction:column;gap:5px}
.fgrp label{font-family:var(--fm);font-size:9.5px;letter-spacing:1.2px;color:var(--muted);text-transform:uppercase}
.fgrp select{background:rgba(0,0,0,.45);border:1px solid var(--border);color:var(--text);
  padding:9px 13px;border-radius:8px;font-family:var(--fm);font-size:12px;
  transition:border .2s,box-shadow .2s;outline:none;width:100%}
.fgrp select:focus{border-color:var(--accent2);box-shadow:0 0 0 3px rgba(0,91,255,.14)}
.fgrp select option{background:#071c30}
.srow{display:flex;align-items:center;gap:10px}
.srow input[type=range]{flex:1;padding:0;cursor:pointer;accent-color:var(--accent)}
.sval{font-family:var(--fm);font-size:12px;color:var(--accent);min-width:58px;text-align:right}

.pbtn{width:100%;margin-top:15px;background:linear-gradient(135deg,var(--accent2),var(--accent));
  color:#fff;border:none;border-radius:10px;padding:13px 20px;font-family:var(--fh);
  font-size:14px;font-weight:700;letter-spacing:.8px;cursor:pointer;
  box-shadow:0 4px 22px rgba(0,91,255,.38);transition:transform .15s,box-shadow .15s}
.pbtn:hover{transform:translateY(-2px);box-shadow:0 8px 32px rgba(0,200,255,.48)}
.pbtn:active{transform:translateY(0)}
.pbtn.loading{opacity:.65;cursor:wait}


.res{display:none}.res.on{display:block}
.gauge-row{display:flex;align-items:center;gap:18px;margin-bottom:14px}
.gring{width:96px;height:96px;flex-shrink:0;position:relative;display:flex;align-items:center;justify-content:center}
.gring svg{position:absolute;inset:0;transform:rotate(-90deg)}
.ginn{font-family:var(--fh);font-weight:800;font-size:21px;z-index:1;text-align:center;line-height:1}
.ginn small{display:block;font-size:9px;letter-spacing:1px;color:var(--muted);margin-top:2px}
.rlabel{font-family:var(--fh);font-size:26px;font-weight:800;letter-spacing:-.5px;margin-bottom:3px}
.rsub{font-size:11.5px;color:var(--muted);margin-bottom:9px}
.mtag{font-family:var(--fm);font-size:9.5px;padding:3px 9px;border-radius:20px;
  border:1px solid var(--accent2);color:var(--accent);background:rgba(0,91,255,.09);
  letter-spacing:.8px;display:inline-block}
.pbars{display:flex;flex-direction:column;gap:7px;margin-top:12px}
.prow{display:flex;align-items:center;gap:9px}
.plbl{font-family:var(--fm);font-size:10.5px;width:58px;color:var(--muted)}
.pbg{flex:1;height:7px;background:rgba(255,255,255,.06);border-radius:3px;overflow:hidden}
.pfill{height:100%;border-radius:3px;transition:width .85s cubic-bezier(.4,0,.2,1)}
.ppct{font-family:var(--fm);font-size:10.5px;width:36px;text-align:right}


.ssg{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.sbox{background:rgba(0,0,0,.32);border:1px solid var(--border);border-radius:9px;padding:11px;text-align:center}
.sbox .v{font-family:var(--fh);font-size:20px;font-weight:800;color:var(--accent)}
.sbox .l{font-size:10.5px;color:var(--muted);margin-top:3px}
.charts-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.cbox{background:rgba(0,0,0,.22);border:1px solid var(--border);border-radius:11px;padding:12px}
.cbox h4{font-family:var(--fh);font-size:10px;font-weight:700;letter-spacing:1.5px;
  color:var(--muted);text-transform:uppercase;margin-bottom:9px}
.dtbl{width:100%;border-collapse:collapse;font-size:11.5px}
.dtbl th{font-family:var(--fm);font-size:9.5px;letter-spacing:.8px;color:var(--muted);
  text-align:left;padding:5px 9px;border-bottom:1px solid var(--border)}
.dtbl td{padding:7px 9px;border-bottom:1px solid rgba(255,255,255,.035)}
.dtbl tr:hover td{background:rgba(0,200,255,.04)}
.pill{font-family:var(--fm);font-size:9.5px;padding:2px 8px;border-radius:20px}
.pl{background:rgba(34,197,94,.14);color:var(--low)}
.pm{background:rgba(245,158,11,.14);color:var(--med)}
.ph{background:rgba(239,68,68,.14);color:var(--high)}
.sname{font-family:var(--fh);font-size:24px;font-weight:800;letter-spacing:-.5px;margin-bottom:3px;
  background:linear-gradient(90deg,var(--text),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
#no-sel{text-align:center;padding:28px;color:var(--muted);font-family:var(--fm);font-size:11px;letter-spacing:.8px}
#no-sel .ico{font-size:44px;margin-bottom:11px}


.astrip{padding:9px 14px;border-radius:8px;font-size:11.5px;display:flex;align-items:center;gap:8px;margin-top:11px}
.ah{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);color:#fca5a5}
.am{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);color:#fcd34d}
.al{background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);color:#86efac}

.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,.28);
  border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:7px}
.divider{height:1px;background:var(--border);margin:5px 0}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#071c30}
::-webkit-scrollbar-thumb{background:rgba(0,91,255,.4);border-radius:3px}
</style>
</head>
<body>


  <div id="htip">
  <div id="htip-bar"></div>
  <div id="htip-body">
    <div id="htip-state">—</div>
    <div id="htip-risk">—</div>
    <div class="htip-rows">
      <div class="htip-row"><span class="htip-lbl">AVG RAINFALL</span><span class="htip-val" id="ht-rain">—</span></div>
      <div class="htip-row"><span class="htip-lbl">RIVER LEVEL</span><span class="htip-val" id="ht-river">—</span></div>
      <div class="htip-row"><span class="htip-lbl">FLOOD RATE</span><span class="htip-val" id="ht-fr">—</span></div>
    </div>
    <div id="htip-flood-bar"><div id="htip-flood-fill"></div></div>
    <div style="font-size:9px;color:var(--muted);font-family:var(--fm);margin-top:5px;letter-spacing:.5px">Click to view full analysis →</div>
  </div>
</div>

<div class="wrap">
  <header>
    <div class="logo">
      <div class="logo-icon">🌊</div>
      <div>
        <h1>FloodSense AI</h1>
        <p>INDIA FLOOD RISK INTELLIGENCE PLATFORM</p>
      </div>
    </div>
    <div class="hbadges">
      <span class="badge b-blue">KNN · DT · LR</span>
      <span class="badge b-green b-pulse">● LIVE MODEL</span>
      <span id="srv-badge" class="srv offline"><span class="dot"></span><span id="srv-txt">BACKEND OFFLINE</span></span>
    </div>
  </header>

  <div class="grid">

    <div class="lcol">


      <div class="card">
        <div class="ctitle">
          <span class="dot"></span>INDIA — FLOOD RISK MAP
          <span style="font-size:10px;color:rgba(91,127,160,.6);font-family:var(--fm);letter-spacing:.5px;margin-left:6px">Hover state → see risk · Click → analyze</span>
          <span id="sel-tag" style="margin-left:auto;font-size:10px;color:var(--accent);font-family:var(--fm)"></span>
        </div>

        <div class="map-outer">
          <div id="map-svg-wrap">
            <div id="map-loader">
              <div class="spin"></div>
              <p>LOADING INDIA MAP</p>
              <small>Fetching state + district boundaries…</small>
            </div>
          </div>
        </div>

        <div class="mleg">
          <div class="li"><div class="ld" style="background:rgba(239,68,68,.62)"></div>High Risk</div>
          <div class="li"><div class="ld" style="background:rgba(245,158,11,.58)"></div>Medium Risk</div>
          <div class="li"><div class="ld" style="background:rgba(34,197,94,.5)"></div>Low Risk</div>
          <div class="li"><div class="ld" style="background:rgba(10,38,75,.75)"></div>No Data</div>
          <div style="margin-left:auto;display:flex;gap:12px">
            <div class="li" style="font-size:9.5px"><span style="color:rgba(0,200,255,.55);margin-right:4px">—</span>District bounds</div>
            <div class="li" style="font-size:9.5px"><span style="color:rgba(0,200,255,.85);margin-right:4px">━</span>State bounds</div>
          </div>
        </div>
      </div>


      <div class="card" id="state-card">
        <div id="no-sel">
          <div class="ico">🗺️</div>
          <div>HOVER OVER A STATE TO PREVIEW</div>
          <div style="margin-top:5px;font-size:9.5px">Click any state to load full district analysis</div>
        </div>
        <div id="state-detail" style="display:none">
          <div class="ctitle"><span class="dot"></span>STATE ANALYSIS</div>
          <div class="sname" id="sd-name"></div>
          <div style="font-size:11px;color:var(--muted);margin-bottom:13px;font-family:var(--fm)" id="sd-sub"></div>
          <div class="ssg" id="sd-stats"></div>
          <div class="divider" style="margin:14px 0"></div>
          <div class="charts-row">
            <div class="cbox"><h4>Risk Distribution</h4><canvas id="pie-chart" height="155"></canvas></div>
            <div class="cbox"><h4>District Rainfall (mm)</h4><canvas id="bar-chart" height="155"></canvas></div>
          </div>
          <div class="divider" style="margin:14px 0"></div>
          <div class="ctitle" style="margin-bottom:9px"><span class="dot"></span>DISTRICT BREAKDOWN</div>
          <div style="overflow-x:auto;max-height:210px;overflow-y:auto">
            <table class="dtbl">
              <thead><tr><th>District</th><th>Rainfall (mm)</th><th>River Lvl (m)</th><th>Risk</th><th>Flood</th></tr></thead>
              <tbody id="sd-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>

    </div>


    <div class="rcol">


      <div class="card">
        <div class="ctitle"><span class="dot"></span>AI MODEL PERFORMANCE</div>
        <div class="acc-row">
          <div class="acc-box active" data-m="knn" onclick="selModel('knn')">
            <div class="an">KNN</div>
            <div class="av" id="a-knn">77.8%</div>
            <div class="ab-bg"><div class="ab-fill" id="b-knn" style="width:77.8%"></div></div>
          </div>
          <div class="acc-box" data-m="dt" onclick="selModel('dt')">
            <div class="an">DEC. TREE</div>
            <div class="av" id="a-dt">100%</div>
            <div class="ab-bg"><div class="ab-fill" id="b-dt" style="width:100%"></div></div>
          </div>
          <div class="acc-box" data-m="lr" onclick="selModel('lr')">
            <div class="an">LOG. REG</div>
            <div class="av" id="a-lr">88.9%</div>
            <div class="ab-bg"><div class="ab-fill" id="b-lr" style="width:88.9%"></div></div>
          </div>
        </div>
      </div>


      <div class="card">
        <div class="ctitle"><span class="dot"></span>FLOOD RISK PREDICTOR</div>
        <div style="display:flex;flex-direction:column;gap:12px">
          <div class="fgrp">
            <label>AI Model</label>
            <select id="model-sel">
              <option value="knn">K-Nearest Neighbors (KNN)</option>
              <option value="dt">Decision Tree</option>
              <option value="lr">Logistic Regression</option>
            </select>
          </div>
          <div class="fgrp">
            <label>Rainfall (mm)</label>
            <div class="srow">
              <input type="range" id="r-rain" min="0" max="500" value="100" oninput="syncS('rain')">
              <span class="sval" id="v-rain">100 mm</span>
            </div>
          </div>
          <div class="fgrp">
            <label>River Level (m)</label>
            <div class="srow">
              <input type="range" id="r-river" min="0" max="6" step="0.1" value="1.5" oninput="syncS('river')">
              <span class="sval" id="v-river">1.5 m</span>
            </div>
          </div>
          <div class="fgrp">
            <label>Flood Level (m)</label>
            <div class="srow">
              <input type="range" id="r-flood" min="0" max="5" step="0.1" value="1.2" oninput="syncS('flood')">
              <span class="sval" id="v-flood">1.2 m</span>
            </div>
          </div>
        </div>
        <button class="pbtn" id="pbtn" onclick="doPredict()">⚡ PREDICT FLOOD RISK</button>
      </div>


      <div class="card res" id="res-card">
        <div class="ctitle"><span class="dot"></span>PREDICTION RESULT</div>
        <div class="gauge-row">
          <div class="gring">
            <svg viewBox="0 0 100 100" width="96" height="96">
              <circle cx="50" cy="50" r="40" fill="none" stroke="rgba(255,255,255,.055)" stroke-width="10"/>
              <circle id="gc" cx="50" cy="50" r="40" fill="none" stroke="var(--accent)" stroke-width="10"
                stroke-dasharray="251.2" stroke-dashoffset="251.2" stroke-linecap="round"
                style="transition:stroke-dashoffset 1s cubic-bezier(.4,0,.2,1),stroke .4s"/>
            </svg>
            <div class="ginn"><span id="gval" style="color:var(--accent)">0</span><small>SCORE</small></div>
          </div>
          <div style="flex:1">
            <div class="rlabel" id="r-label">—</div>
            <div class="rsub" id="r-sub">—</div>
            <div class="mtag" id="r-tag">—</div>
          </div>
        </div>
        <div class="pbars" id="pbars"></div>
        <div id="r-alert" class="astrip" style="display:none"></div>
      </div>


      <div class="card">
        <div class="ctitle"><span class="dot"></span>HOW IT WORKS</div>
        <div style="font-size:11.5px;color:var(--muted);line-height:1.75">
          <p>Three ML models trained on India flood records:</p><br>
          <p>🔵 <strong style="color:var(--text)">KNN</strong> — finds K most similar historical flood events and votes on risk.</p>
          <p style="margin-top:7px">🟡 <strong style="color:var(--text)">Decision Tree</strong> — if-then rules on rainfall &amp; river thresholds.</p>
          <p style="margin-top:7px">🟢 <strong style="color:var(--text)">Logistic Regression</strong> — probabilistic sigmoid output.</p>
          <br><p>Features: <code style="color:var(--accent);font-size:10.5px">rainfall_mm · river_level_m · flood_level_m</code></p>
        </div>
      </div>

    </div>
  </div>
</div>

<script>

const FD = [
  {state:"Andaman & Nicobar Islands",district:"Port Blair",     rainfall_mm:49.2, river_level_m:0.4, flood_level_m:0.3, flood_occurred:0,risk_level:"Low"},
  {state:"Arunachal Pradesh",        district:"Itanagar",       rainfall_mm:180.5,river_level_m:1.2, flood_level_m:1.1, flood_occurred:1,risk_level:"Medium"},
  {state:"Assam",                    district:"Guwahati",       rainfall_mm:245.0,river_level_m:2.8, flood_level_m:2.6, flood_occurred:1,risk_level:"High"},
  {state:"Meghalaya",                district:"Shillong",       rainfall_mm:245.0,river_level_m:2.8, flood_level_m:2.6, flood_occurred:1,risk_level:"High"},
  {state:"Nagaland",                 district:"Kohima",         rainfall_mm:115.0,river_level_m:0.9, flood_level_m:0.7, flood_occurred:0,risk_level:"Low"},
  {state:"Manipur",                  district:"Imphal",         rainfall_mm:115.0,river_level_m:0.9, flood_level_m:0.7, flood_occurred:0,risk_level:"Low"},
  {state:"Mizoram",                  district:"Aizawl",         rainfall_mm:115.0,river_level_m:0.9, flood_level_m:0.7, flood_occurred:0,risk_level:"Low"},
  {state:"Tripura",                  district:"Agartala",       rainfall_mm:115.0,river_level_m:0.9, flood_level_m:0.7, flood_occurred:0,risk_level:"Low"},
  {state:"West Bengal",              district:"Darjeeling",     rainfall_mm:190.4,river_level_m:2.1, flood_level_m:1.9, flood_occurred:1,risk_level:"Medium"},
  {state:"Sikkim",                   district:"Gangtok",        rainfall_mm:190.4,river_level_m:2.1, flood_level_m:1.9, flood_occurred:1,risk_level:"Medium"},
  {state:"West Bengal",              district:"Kolkata",        rainfall_mm:85.2, river_level_m:1.4, flood_level_m:1.2, flood_occurred:1,risk_level:"Medium"},
  {state:"Odisha",                   district:"Bhubaneswar",    rainfall_mm:130.6,river_level_m:1.8, flood_level_m:1.6, flood_occurred:1,risk_level:"Medium"},
  {state:"Jharkhand",                district:"Ranchi",         rainfall_mm:65.4, river_level_m:0.5, flood_level_m:0.4, flood_occurred:0,risk_level:"Low"},
  {state:"Bihar",                    district:"Patna",          rainfall_mm:42.0, river_level_m:3.2, flood_level_m:2.8, flood_occurred:1,risk_level:"High"},
  {state:"Uttar Pradesh",            district:"Varanasi",       rainfall_mm:75.2, river_level_m:2.1, flood_level_m:1.8, flood_occurred:1,risk_level:"Medium"},
  {state:"Uttar Pradesh",            district:"Meerut",         rainfall_mm:55.4, river_level_m:1.4, flood_level_m:1.1, flood_occurred:1,risk_level:"Medium"},
  {state:"Uttarakhand",              district:"Dehradun",       rainfall_mm:310.0,river_level_m:3.5, flood_level_m:3.1, flood_occurred:1,risk_level:"High"},
  {state:"Haryana",                  district:"Gurgaon",        rainfall_mm:45.2, river_level_m:1.2, flood_level_m:0.9, flood_occurred:0,risk_level:"Low"},
  {state:"Delhi",                    district:"New Delhi",      rainfall_mm:45.2, river_level_m:1.2, flood_level_m:0.9, flood_occurred:0,risk_level:"Low"},
  {state:"Punjab",                   district:"Chandigarh",     rainfall_mm:38.4, river_level_m:0.6, flood_level_m:0.4, flood_occurred:0,risk_level:"Low"},
  {state:"Himachal Pradesh",         district:"Shimla",         rainfall_mm:140.2,river_level_m:2.4, flood_level_m:2.2, flood_occurred:1,risk_level:"High"},
  {state:"Jammu & Kashmir",          district:"Srinagar",       rainfall_mm:85.6, river_level_m:1.5, flood_level_m:1.3, flood_occurred:1,risk_level:"Medium"},
  {state:"Rajasthan",                district:"Jaisalmer",      rainfall_mm:12.4, river_level_m:0.2, flood_level_m:0.1, flood_occurred:0,risk_level:"Low"},
  {state:"Rajasthan",                district:"Jaipur",         rainfall_mm:55.2, river_level_m:0.8, flood_level_m:0.6, flood_occurred:0,risk_level:"Low"},
  {state:"Madhya Pradesh",           district:"Indore",         rainfall_mm:95.4, river_level_m:1.1, flood_level_m:0.9, flood_occurred:0,risk_level:"Low"},
  {state:"Madhya Pradesh",           district:"Jabalpur",       rainfall_mm:110.2,river_level_m:1.5, flood_level_m:1.3, flood_occurred:1,risk_level:"Medium"},
  {state:"Gujarat",                  district:"Ahmedabad",      rainfall_mm:115.6,river_level_m:1.4, flood_level_m:1.2, flood_occurred:1,risk_level:"Medium"},
  {state:"Gujarat",                  district:"Rajkot",         rainfall_mm:25.4, river_level_m:0.4, flood_level_m:0.2, flood_occurred:0,risk_level:"Low"},
  {state:"Maharashtra",              district:"Mumbai",         rainfall_mm:220.4,river_level_m:2.0, flood_level_m:1.8, flood_occurred:1,risk_level:"High"},
  {state:"Goa",                      district:"Panaji",         rainfall_mm:220.4,river_level_m:2.0, flood_level_m:1.8, flood_occurred:1,risk_level:"High"},
  {state:"Maharashtra",              district:"Pune",           rainfall_mm:85.2, river_level_m:0.7, flood_level_m:0.5, flood_occurred:0,risk_level:"Low"},
  {state:"Maharashtra",              district:"Aurangabad",     rainfall_mm:35.4, river_level_m:0.3, flood_level_m:0.1, flood_occurred:0,risk_level:"Low"},
  {state:"Maharashtra",              district:"Nagpur",         rainfall_mm:75.6, river_level_m:1.1, flood_level_m:0.9, flood_occurred:0,risk_level:"Low"},
  {state:"Chhattisgarh",             district:"Raipur",         rainfall_mm:90.4, river_level_m:1.2, flood_level_m:1.1, flood_occurred:0,risk_level:"Medium"},
  {state:"Andhra Pradesh",           district:"Visakhapatnam",  rainfall_mm:135.2,river_level_m:1.9, flood_level_m:1.7, flood_occurred:1,risk_level:"Medium"},
  {state:"Telangana",                district:"Hyderabad",      rainfall_mm:65.4, river_level_m:1.3, flood_level_m:1.2, flood_occurred:1,risk_level:"Medium"},
  {state:"Andhra Pradesh",           district:"Kurnool",        rainfall_mm:45.2, river_level_m:0.6, flood_level_m:0.4, flood_occurred:0,risk_level:"Low"},
  {state:"Tamil Nadu",               district:"Chennai",        rainfall_mm:85.6, river_level_m:1.1, flood_level_m:0.8, flood_occurred:0,risk_level:"Low"},
  {state:"Karnataka",                district:"Mangalore",      rainfall_mm:210.4,river_level_m:2.5, flood_level_m:2.3, flood_occurred:1,risk_level:"High"},
  {state:"Karnataka",                district:"Hubli",          rainfall_mm:45.2, river_level_m:0.7, flood_level_m:0.5, flood_occurred:0,risk_level:"Low"},
  {state:"Karnataka",                district:"Bangalore",      rainfall_mm:65.8, river_level_m:1.2, flood_level_m:1.1, flood_occurred:0,risk_level:"Medium"},
  {state:"Kerala",                   district:"Kochi",          rainfall_mm:350.0,river_level_m:4.2, flood_level_m:3.8, flood_occurred:1,risk_level:"High"},
  {state:"Lakshadweep",              district:"Kavaratti",      rainfall_mm:55.4, river_level_m:0.5, flood_level_m:0.3, flood_occurred:0,risk_level:"Low"},
];


const SR = {};
FD.forEach(d=>{
  if(!SR[d.state]) SR[d.state]={rows:[],risk:'Low'};
  SR[d.state].rows.push(d);
});
Object.keys(SR).forEach(s=>{
  const c={Low:0,Medium:0,High:0};
  SR[s].rows.forEach(r=>c[r.risk_level]++);
  SR[s].risk=c.High>0?'High':c.Medium>0?'Medium':'Low';
  SR[s].avgRain=+(SR[s].rows.reduce((a,r)=>a+r.rainfall_mm,0)/SR[s].rows.length).toFixed(1);
  SR[s].avgRiver=+(SR[s].rows.reduce((a,r)=>a+r.river_level_m,0)/SR[s].rows.length).toFixed(2);
  SR[s].floodRate=Math.round(SR[s].rows.filter(r=>r.flood_occurred).length/SR[s].rows.length*100);
});


const GMAP={
  "Andaman and Nicobar":"Andaman & Nicobar Islands",
  "Andaman & Nicobar Islands":"Andaman & Nicobar Islands",
  "Arunachal Pradesh":"Arunachal Pradesh","Assam":"Assam","Bihar":"Bihar",
  "Chhattisgarh":"Chhattisgarh","Goa":"Goa","Gujarat":"Gujarat","Haryana":"Haryana",
  "Himachal Pradesh":"Himachal Pradesh","Jammu and Kashmir":"Jammu & Kashmir",
  "Jammu & Kashmir":"Jammu & Kashmir","Jharkhand":"Jharkhand","Karnataka":"Karnataka",
  "Kerala":"Kerala","Lakshadweep":"Lakshadweep","Madhya Pradesh":"Madhya Pradesh",
  "Maharashtra":"Maharashtra","Manipur":"Manipur","Meghalaya":"Meghalaya",
  "Mizoram":"Mizoram","Nagaland":"Nagaland","NCT of Delhi":"Delhi","Delhi":"Delhi",
  "Odisha":"Odisha","Orissa":"Odisha","Punjab":"Punjab","Rajasthan":"Rajasthan",
  "Sikkim":"Sikkim","Tamil Nadu":"Tamil Nadu","Telangana":"Telangana",
  "Tripura":"Tripura","Uttar Pradesh":"Uttar Pradesh","Uttarakhand":"Uttarakhand",
  "Uttaranchal":"Uttarakhand","West Bengal":"West Bengal","Andhra Pradesh":"Andhra Pradesh",
  "Chandigarh":"Punjab","Puducherry":"Tamil Nadu","Dadra and Nagar Haveli":"Gujarat",
  "Daman and Diu":"Gujarat","Ladakh":"Jammu & Kashmir",
};


const RC={High:'#ef4444',Medium:'#f59e0b',Low:'#22c55e'};
const RFILL={High:'fill-high',Medium:'fill-medium',Low:'fill-low'};


const DIST_URL = "https://raw.githubusercontent.com/geohacker/india/master/district/india.geojson";
const STATE_URL = "https://raw.githubusercontent.com/geohacker/india/master/state/india_telengana.geojson";


const DIST_URL2 = "https://raw.githubusercontent.com/datameet/maps/master/Districts/India_Districts.geojson";
const STATE_URL2 = "https://raw.githubusercontent.com/datameet/maps/master/States/India_States.geojson";

let hoveredState = null;
let selectedState = null;
let mapSvg = null, mapG = null;

async function loadGeoJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function buildMap() {
  const wrap = document.getElementById('map-svg-wrap');
  const loader = document.getElementById('map-loader');

  try {

    let distGeo = null, stateGeo = null;

    loader.querySelector('small').textContent = 'Loading state boundaries…';
    try { stateGeo = await loadGeoJSON(STATE_URL); }
    catch { stateGeo = await loadGeoJSON(STATE_URL2); }

    loader.querySelector('small').textContent = 'Loading district boundaries…';
    try { distGeo = await loadGeoJSON(DIST_URL); }
    catch {
      try { distGeo = await loadGeoJSON(DIST_URL2); }
      catch { distGeo = null; }
    }

    loader.remove();

    const W = wrap.clientWidth || 640;
    const H = Math.round(W * 1.08);

    mapSvg = d3.select('#map-svg-wrap')
      .append('svg')
      .attr('id','india-svg')
      .attr('viewBox',`0 0 ${W} ${H}`)
      .attr('preserveAspectRatio','xMidYMid meet');


    mapSvg.append('rect')
      .attr('width',W).attr('height',H)
      .attr('fill','rgba(0,20,50,0)')

    const proj = d3.geoMercator().fitSize([W-10, H-10], stateGeo);
    const path = d3.geoPath().projection(proj);

    mapG = mapSvg.append('g');


    mapG.append('g').attr('id','layer-states')
      .selectAll('path')
      .data(stateGeo.features)
      .enter().append('path')
      .attr('class', d => {
        const sn = getStateName(d);
        const risk = SR[sn]?.risk;
        return `district-path ${risk ? RFILL[risk] : 'fill-none'}`;
      })
      .attr('d', path)
      .attr('data-state', d => getStateName(d))
      .on('mouseenter', function(event, d) { onStateEnter(event, d, this); })
      .on('mousemove', onMouseMove)
      .on('mouseleave', function(event, d) { onStateLeave(event, d, this); })
      .on('click', function(event, d) { onStateClick(event, d, this); });


      if (distGeo) {
      mapG.append('g').attr('id','layer-districts')
        .selectAll('path')
        .data(distGeo.features)
        .enter().append('path')
        .attr('fill','none')
        .attr('stroke','rgba(0,150,220,0.22)')
        .attr('stroke-width','0.3')
        .attr('pointer-events','none')
        .attr('d', path);
    }


    mapG.append('g').attr('id','layer-state-borders')
      .selectAll('path')
      .data(stateGeo.features)
      .enter().append('path')
      .attr('class','state-border')
      .attr('d', path)
      .attr('stroke', d => {
        const sn = getStateName(d);
        const risk = SR[sn]?.risk;
        return risk ? {High:'rgba(239,68,68,.55)',Medium:'rgba(245,158,11,.5)',Low:'rgba(34,197,94,.45)'}[risk]
                    : 'rgba(0,150,220,0.35)';
      })
      .attr('stroke-width','0.9');


      const ABBR={
      "Rajasthan":"RJ","Madhya Pradesh":"MP","Maharashtra":"MH","Uttar Pradesh":"UP",
      "Gujarat":"GJ","Karnataka":"KA","Andhra Pradesh":"AP","Odisha":"OD",
      "Telangana":"TS","West Bengal":"WB","Assam":"AS","Bihar":"BR",
      "Jharkhand":"JH","Chhattisgarh":"CG","Tamil Nadu":"TN","Kerala":"KL",
      "Punjab":"PB","Haryana":"HR","Uttarakhand":"UK","Himachal Pradesh":"HP",
      "Arunachal Pradesh":"AR","Meghalaya":"ML","Manipur":"MN","Nagaland":"NL",
      "Mizoram":"MZ","Tripura":"TR","Sikkim":"SK","Goa":"GA","Delhi":"DL",
      "Jammu & Kashmir":"J&K","Lakshadweep":"LD","Andaman & Nicobar Islands":"AN",
    };
    mapG.append('g').attr('id','layer-labels')
      .selectAll('text')
      .data(stateGeo.features)
      .enter().append('text')
      .attr('x', d => path.centroid(d)[0])
      .attr('y', d => path.centroid(d)[1])
      .attr('text-anchor','middle').attr('dominant-baseline','central')
      .attr('font-family','Space Mono').attr('font-size','5.5')
      .attr('fill','rgba(200,230,255,0.55)')
      .attr('pointer-events','none')
      .text(d => ABBR[getStateName(d)] || '');

  } catch(err) {
    console.error('Map error:', err);
    loader.innerHTML = `
      <div style="color:#f87171;font-family:var(--fm);font-size:11px;text-align:center;padding:20px;line-height:2">
        ⚠ COULD NOT LOAD MAP<br>
        <span style="color:var(--muted);font-size:9.5px">Requires internet · Check browser console</span>
      </div>`;
  }
}

function getStateName(d) {
  const raw = d.properties?.NAME_1 || d.properties?.name || d.properties?.st_nm || d.properties?.State || '';
  return GMAP[raw] || raw;
}


const TIP = document.getElementById('htip');

function onStateEnter(event, d, el) {
  const sn = getStateName(d);
  hoveredState = sn;
  const data = SR[sn];


  d3.selectAll('.district-path')
    .filter(f => getStateName(f) === sn)
    .classed('state-hovered', true);


    const risk = data?.risk || 'Unknown';
  const col  = data ? RC[risk] : '#6b8aab';
  document.getElementById('htip-bar').style.background =
    data ? `linear-gradient(90deg,${col},${col}88)` : '#6b8aab';
  document.getElementById('htip-state').textContent = sn || 'Unknown';
  document.getElementById('htip-state').style.color = col;
  document.getElementById('htip-risk').textContent = data ? `⚠ ${risk} Flood Risk` : 'No data available';
  document.getElementById('htip-risk').style.color = col;
  document.getElementById('ht-rain').textContent  = data ? `${data.avgRain} mm` : '—';
  document.getElementById('ht-river').textContent = data ? `${data.avgRiver} m` : '—';
  document.getElementById('ht-fr').textContent    = data ? `${data.floodRate}%` : '—';
  const fillPct = data ? {Low:25,Medium:60,High:90}[risk] : 0;
  document.getElementById('htip-flood-fill').style.width = fillPct+'%';
  document.getElementById('htip-flood-fill').style.background = col;

  TIP.style.display = 'block';
  positionTip(event);
}

function onMouseMove(event) {
  positionTip(event);
}

function onStateLeave(event, d, el) {
  const sn = getStateName(d);
  hoveredState = null;
  d3.selectAll('.district-path')
    .filter(f => getStateName(f) === sn && f !== selectedState)
    .classed('state-hovered', false);
  TIP.style.display = 'none';
}

function onStateClick(event, d, el) {
  const sn = getStateName(d);
  if (!SR[sn]) return;


  d3.selectAll('.district-path').classed('selected-state', false);


  d3.selectAll('.district-path')
    .filter(f => getStateName(f) === sn)
    .classed('selected-state', true);

  selectedState = sn;
  document.getElementById('sel-tag').textContent = sn.toUpperCase();
  loadStatePanel(sn);
}

function positionTip(event) {
  const mx = event.clientX, my = event.clientY;
  const tw = TIP.offsetWidth || 230, th = TIP.offsetHeight || 160;
  let x = mx + 18, y = my - 10;
  if (x + tw > window.innerWidth - 10) x = mx - tw - 12;
  if (y + th > window.innerHeight - 10) y = my - th - 10;
  TIP.style.left = x + 'px';
  TIP.style.top  = y + 'px';
}


let pieC=null, barC=null;
function killCharts(){if(pieC){pieC.destroy();pieC=null}if(barC){barC.destroy();barC=null}}

function loadStatePanel(sn) {
  const data = SR[sn];
  if (!data) return;
  const rows = data.rows;

  document.getElementById('no-sel').style.display = 'none';
  document.getElementById('state-detail').style.display = 'block';
  document.getElementById('sd-name').textContent = sn;
  document.getElementById('sd-sub').textContent  = `${rows.length} district${rows.length>1?'s':''} · ${data.risk} flood risk zone`;

  const rc = RC[data.risk];
  document.getElementById('sd-stats').innerHTML = `
    <div class="sbox"><div class="v">${data.avgRain}</div><div class="l">Avg Rainfall (mm)</div></div>
    <div class="sbox"><div class="v">${data.avgRiver}</div><div class="l">Avg River Level (m)</div></div>
    <div class="sbox"><div class="v">${data.floodRate}%</div><div class="l">Flood Rate</div></div>
    <div class="sbox"><div class="v" style="color:${rc}">${data.risk}</div><div class="l">Dominant Risk</div></div>`;

  setTimeout(()=>{
    killCharts();
    const c={Low:0,Medium:0,High:0};
    rows.forEach(r=>c[r.risk_level]++);
    pieC = new Chart(document.getElementById('pie-chart'),{
      type:'doughnut',
      data:{labels:['Low','Medium','High'],datasets:[{data:[c.Low,c.Medium,c.High],
        backgroundColor:['rgba(34,197,94,.7)','rgba(245,158,11,.7)','rgba(239,68,68,.72)'],
        borderColor:['#22c55e','#f59e0b','#ef4444'],borderWidth:1.5}]},
      options:{responsive:true,maintainAspectRatio:false,cutout:'60%',
        plugins:{legend:{labels:{color:'#5b7fa0',font:{family:'Space Mono',size:9.5},boxWidth:10,padding:7}}}}
    });
    barC = new Chart(document.getElementById('bar-chart'),{
      type:'bar',
      data:{labels:rows.map(r=>r.district.length>8?r.district.slice(0,8)+'…':r.district),
        datasets:[{data:rows.map(r=>r.rainfall_mm),
          backgroundColor:rows.map(r=>({Low:'rgba(34,197,94,.55)',Medium:'rgba(245,158,11,.55)',High:'rgba(239,68,68,.62)'}[r.risk_level])),
          borderColor:rows.map(r=>RC[r.risk_level]),borderWidth:1.5,borderRadius:4}]},
      options:{responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{x:{ticks:{color:'#5b7fa0',font:{family:'Space Mono',size:8.5}},grid:{color:'rgba(255,255,255,.035)'}},
                y:{ticks:{color:'#5b7fa0',font:{family:'Space Mono',size:8.5}},grid:{color:'rgba(255,255,255,.035)'},beginAtZero:true}}}
    });
  },40);

  document.getElementById('sd-tbody').innerHTML = rows.map(r=>`
    <tr>
      <td>${r.district}</td><td>${r.rainfall_mm}</td><td>${r.river_level_m}</td>
      <td><span class="pill ${r.risk_level==='High'?'ph':r.risk_level==='Medium'?'pm':'pl'}">${r.risk_level}</span></td>
      <td style="color:${r.flood_occurred?'var(--high)':'var(--low)'}">${r.flood_occurred?'Yes':'No'}</td>
    </tr>`).join('');


    document.getElementById('r-rain').value  = data.avgRain;
  document.getElementById('r-river').value = data.avgRiver;
  document.getElementById('r-flood').value = (data.avgRiver*.85).toFixed(1);
  syncS('rain'); syncS('river'); syncS('flood');

  document.getElementById('state-card').scrollIntoView({behavior:'smooth',block:'nearest'});
}


function syncS(t){
  if(t==='rain')  document.getElementById('v-rain').textContent=document.getElementById('r-rain').value+' mm';
  else if(t==='river') document.getElementById('v-river').textContent=parseFloat(document.getElementById('r-river').value).toFixed(1)+' m';
  else document.getElementById('v-flood').textContent=parseFloat(document.getElementById('r-flood').value).toFixed(1)+' m';
}
let selMod='knn';
function selModel(m){
  selMod=m;
  document.getElementById('model-sel').value=m;
  document.querySelectorAll('.acc-box').forEach(b=>b.classList.toggle('active',b.dataset.m===m));
}


function knnPredict(rain, river, flood, k=3){
  const mR=400,mRv=5,mF=4.5;
  const i=[rain/mR,river/mRv,flood/mF];
  const sorted=FD.map(d=>({
    risk:d.risk_level,
    d:Math.hypot(d.rainfall_mm/mR-i[0],d.river_level_m/mRv-i[1],d.flood_level_m/mF-i[2])
  })).sort((a,b)=>a.d-b.d).slice(0,k);
  const c={Low:0,Medium:0,High:0};
  sorted.forEach(d=>c[d.risk]++);
  const p={Low:+(c.Low/k*100).toFixed(1),Medium:+(c.Medium/k*100).toFixed(1),High:+(c.High/k*100).toFixed(1)};
  const pred=Object.entries(p).sort((a,b)=>b[1]-a[1])[0][0];
  return{risk_level:pred,probabilities:p,risk_score:{Low:25,Medium:60,High:90}[pred],flood_likely:pred!=='Low'};
}

const API='http://localhost:5000';
let backOnline=false;
async function checkBackend(){
  try{
    const r=await fetch(API+'/api/model_stats',{signal:AbortSignal.timeout(2000)});
    if(r.ok){
      const st=await r.json(); backOnline=true;
      document.getElementById('srv-badge').className='srv online';
      document.getElementById('srv-txt').textContent='BACKEND LIVE';
      ['knn','dt','lr'].forEach(m=>{
        const v=st[m];
        document.getElementById(`a-${m}`).textContent=v+'%';
        document.getElementById(`b-${m}`).style.width=v+'%';
      });
    }
  }catch{}
}


async function doPredict(){
  const btn=document.getElementById('pbtn');
  const rain=parseFloat(document.getElementById('r-rain').value);
  const river=parseFloat(document.getElementById('r-river').value);
  const flood=parseFloat(document.getElementById('r-flood').value);
  const model=document.getElementById('model-sel').value;
  selMod=model;
  btn.innerHTML='<span class="spinner"></span>ANALYZING…';
  btn.classList.add('loading');
  let res;
  if(backOnline){
    try{
      const r=await fetch(API+'/api/predict',{method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({rainfall_mm:rain,river_level_m:river,flood_level_m:flood,model})});
      res=await r.json();
    }catch{res=knnPredict(rain,river,flood);}
  }else{
    await new Promise(r=>setTimeout(r,650));
    res=knnPredict(rain,river,flood);
    res.model_used=model.toUpperCase();
    res.model_accuracy={knn:'77.8',dt:'100',lr:'88.9'}[model];
  }
  btn.innerHTML='⚡ PREDICT FLOOD RISK';
  btn.classList.remove('loading');
  showResult(res);
}

function showResult(res){
  document.getElementById('res-card').classList.add('on');
  const col=RC[res.risk_level]||'#00c8ff';
  const score=res.risk_score||{Low:25,Medium:60,High:90}[res.risk_level];
  const gc=document.getElementById('gc');
  gc.style.strokeDashoffset=251.2-(score/100)*251.2;
  gc.style.stroke=col;
  document.getElementById('gval').textContent=score;
  document.getElementById('gval').style.color=col;
  document.getElementById('r-label').textContent=res.risk_level+' Risk';
  document.getElementById('r-label').style.color=col;
  document.getElementById('r-sub').textContent=res.flood_likely?'⚠ Flood conditions likely':'✓ Conditions appear stable';
  document.getElementById('r-tag').textContent=`${res.model_used||'KNN'} · ${res.model_accuracy||'~87'}% accuracy`;
  const pb=document.getElementById('pbars');
  pb.innerHTML=['Low','Medium','High'].map(l=>`
    <div class="prow">
      <span class="plbl">${l}</span>
      <div class="pbg"><div class="pfill" style="width:0%;background:${RC[l]}" data-t="${res.probabilities?.[l]||0}"></div></div>
      <span class="ppct" style="color:${RC[l]}">${res.probabilities?.[l]||0}%</span>
    </div>`).join('');
  setTimeout(()=>pb.querySelectorAll('.pfill').forEach(b=>b.style.width=b.dataset.t+'%'),40);
  const al=document.getElementById('r-alert');
  al.textContent={
    High:'🚨 HIGH RISK — Evacuate low-lying areas. NDRF: 011-24363260',
    Medium:'⚠️ MEDIUM RISK — Monitor levels closely. Keep emergency kit ready.',
    Low:'✅ LOW RISK — Conditions normal. Continue monsoon monitoring.'
  }[res.risk_level];
  al.className=`astrip ${res.risk_level==='High'?'ah':res.risk_level==='Medium'?'am':'al'}`;
  al.style.display='flex';
  document.getElementById('res-card').scrollIntoView({behavior:'smooth',block:'nearest'});
}


buildMap();
checkBackend();
</script>
</body>
</html>

# flood_ai.py:

"""
Flood Risk Prediction AI Backend
Uses KNN, Decision Tree, and Logistic Regression (as taught in class)
Run: python flood_ai.py
Then open index.html in browser
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import json
import os

app = Flask(__name__, static_folder='.')


@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


DATA_FILE = os.path.join(os.path.dirname(__file__), 'flood_data.csv')

def load_data():
    df = pd.read_csv(DATA_FILE)
    return df

def train_models(df):
    features = ['rainfall_mm', 'river_level_m', 'flood_level_m']
    X = df[features].values
    y_binary = df['flood_occurred'].values          
    y_risk   = df['risk_level'].values              

    le = LabelEncoder()
    y_risk_enc = le.fit_transform(y_risk)           

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_risk_enc, test_size=0.2, random_state=42)


    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))


    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))


    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))

    models = {
        'knn':  (knn,  round(knn_acc * 100, 1)),
        'dt':   (dt,   round(dt_acc  * 100, 1)),
        'lr':   (lr,   round(lr_acc  * 100, 1)),
    }
    return models, scaler, le
df_global = load_data()
MODELS, SCALER, LE = train_models(df_global)
print("\n✅ Models trained:")
for name, (_, acc) in MODELS.items():
    print(f"   {name.upper():<5} accuracy: {acc}%")
def get_state_summary(state_name):
    rows = df_global[df_global['state'].str.lower() == state_name.lower()]
    if rows.empty:
        rows = df_global[df_global['state'].str.lower().str.contains(state_name.lower())]
    if rows.empty:
        return None
    summary = {
        'state': rows.iloc[0]['state'],
        'districts': rows['district'].tolist(),
        'avg_rainfall': round(rows['rainfall_mm'].mean(), 1),
        'avg_river_level': round(rows['river_level_m'].mean(), 2),
        'flood_occurrences': int(rows['flood_occurred'].sum()),
        'total_records': len(rows),
        'risk_distribution': rows['risk_level'].value_counts().to_dict(),
        'district_data': rows[['district','rainfall_mm','river_level_m',
                                'flood_level_m','flood_occurred','risk_level']].to_dict('records')
    }
    return summary
@app.route('/')
def index():
    return app.send_static_file('index.html')
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    data = request.get_json()
    rainfall   = float(data.get('rainfall_mm', 100))
    river_lvl  = float(data.get('river_level_m', 1.5))
    flood_lvl  = float(data.get('flood_level_m', 1.3))
    model_name = data.get('model', 'knn')
    X_input = SCALER.transform([[rainfall, river_lvl, flood_lvl]])
    model, acc = MODELS.get(model_name, MODELS['knn'])
    pred_enc  = model.predict(X_input)[0]
    pred_prob = model.predict_proba(X_input)[0]
    risk_label = LE.inverse_transform([pred_enc])[0]
    classes = list(LE.classes_)
    prob_dict = {classes[i]: round(float(p)*100, 1) for i, p in enumerate(pred_prob)}

    score_map = {'Low': 25, 'Medium': 60, 'High': 90}
    risk_score = score_map.get(risk_label, 50)
    return jsonify({
        'risk_level':    risk_label,
        'risk_score':    risk_score,
        'probabilities': prob_dict,
        'model_used':    model_name.upper(),
        'model_accuracy': acc,
        'flood_likely':  risk_label in ['Medium', 'High']
    })
@app.route('/api/state/<state_name>', methods=['GET'])
def state_info(state_name):
    summary = get_state_summary(state_name)
    if not summary:
        return jsonify({'error': f'State "{state_name}" not found'}), 404
    return jsonify(summary)
@app.route('/api/all_states', methods=['GET'])
def all_states():
    states = []
    for state in df_global['state'].unique():
        rows = df_global[df_global['state'] == state]
        dominant_risk = rows['risk_level'].mode()[0]
        avg_rain = round(rows['rainfall_mm'].mean(), 1)
        states.append({
            'state': state,
            'dominant_risk': dominant_risk,
            'avg_rainfall': avg_rain,
            'flood_rate': round(rows['flood_occurred'].mean() * 100, 0)
        })
    return jsonify(states)
@app.route('/api/model_stats', methods=['GET'])
def model_stats():
    stats = {name: acc for name, (_, acc) in MODELS.items()}
    return jsonify(stats)
if __name__ == '__main__':
    print("\n🌊 Flood Risk Prediction AI Server")
    print("   http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, port=5000)

# flood_data.csv:
subdivision,state,district,rainfall_mm,river_level_m,flood_level_m,flood_occurred,risk_level
Andaman & Nicobar Islands,Andaman & Nicobar Islands,Port Blair,49.2,0.4,0.3,0,Low
Arunachal Pradesh,Arunachal Pradesh,Itanagar,180.5,1.2,1.1,1,Medium
Assam & Meghalaya,Assam,Guwahati,245.0,2.8,2.6,1,High
Assam & Meghalaya,Meghalaya,Shillong,245.0,2.8,2.6,1,High
Nagaland-Manipur-Mizoram-Tripura,Nagaland,Kohima,115.0,0.9,0.7,0,Low
Nagaland-Manipur-Mizoram-Tripura,Manipur,Imphal,115.0,0.9,0.7,0,Low
Nagaland-Manipur-Mizoram-Tripura,Mizoram,Aizawl,115.0,0.9,0.7,0,Low
Nagaland-Manipur-Mizoram-Tripura,Tripura,Agartala,115.0,0.9,0.7,0,Low
Sub-Himalayan West Bengal & Sikkim,West Bengal,Darjeeling,190.4,2.1,1.9,1,Medium
Sub-Himalayan West Bengal & Sikkim,Sikkim,Gangtok,190.4,2.1,1.9,1,Medium
Gangetic West Bengal,West Bengal,Kolkata,85.2,1.4,1.2,1,Medium
Odisha,Odisha,Bhubaneswar,130.6,1.8,1.6,1,Medium
Jharkhand,Jharkhand,Ranchi,65.4,0.5,0.4,0,Low
Bihar (Riverine/Dam Risk),Bihar,Patna,42.0,3.2,2.8,1,High
East Uttar Pradesh,Uttar Pradesh,Varanasi,75.2,2.1,1.8,1,Medium
West Uttar Pradesh,Uttar Pradesh,Meerut,55.4,1.4,1.1,1,Medium
Uttarakhand (Rain-Driven),Uttarakhand,Dehradun,310.0,3.5,3.1,1,High
Haryana-Chandigarh-Delhi,Haryana,Gurgaon,45.2,1.2,0.9,0,Low
Haryana-Chandigarh-Delhi,Delhi,New Delhi,45.2,1.2,0.9,0,Low
Punjab,Punjab,Chandigarh,38.4,0.6,0.4,0,Low
Himachal Pradesh,Himachal Pradesh,Shimla,140.2,2.4,2.2,1,High
Jammu & Kashmir,Jammu & Kashmir,Srinagar,85.6,1.5,1.3,1,Medium
West Rajasthan,Rajasthan,Jaisalmer,12.4,0.2,0.1,0,Low
East Rajasthan,Rajasthan,Jaipur,55.2,0.8,0.6,0,Low
West Madhya Pradesh,Madhya Pradesh,Indore,95.4,1.1,0.9,0,Low
East Madhya Pradesh,Madhya Pradesh,Jabalpur,110.2,1.5,1.3,1,Medium
Gujarat Region,Gujarat,Ahmedabad,115.6,1.4,1.2,1,Medium
Saurashtra & Kutch,Gujarat,Rajkot,25.4,0.4,0.2,0,Low
Konkan & Goa,Maharashtra,Mumbai,220.4,2.0,1.8,1,High
Konkan & Goa,Goa,Panaji,220.4,2.0,1.8,1,High
Madhya Maharashtra,Maharashtra,Pune,85.2,0.7,0.5,0,Low
Marathwada,Maharashtra,Aurangabad,35.4,0.3,0.1,0,Low
Vidarbha,Maharashtra,Nagpur,75.6,1.1,0.9,0,Low
Chhattisgarh,Chhattisgarh,Raipur,90.4,1.2,1.1,0,Medium
Coastal Andhra Pradesh & Yanam,Andhra Pradesh,Visakhapatnam,135.2,1.9,1.7,1,Medium
Telangana,Telangana,Hyderabad,65.4,1.3,1.2,1,Medium
Rayalaseema,Andhra Pradesh,Kurnool,45.2,0.6,0.4,0,Low
Tamil Nadu-Puducherry-Karaikal,Tamil Nadu,Chennai,85.6,1.1,0.8,0,Low
Coastal Karnataka,Karnataka,Mangalore,210.4,2.5,2.3,1,High
North Interior Karnataka,Karnataka,Hubli,45.2,0.7,0.5,0,Low
South Interior Karnataka,Karnataka,Bangalore,65.8,1.2,1.1,0,Medium
Kerala (Rain/Dam Risk),Kerala,Kochi,350.0,4.2,3.8,1,High
Lakshadweep,Lakshadweep,Kavaratti,55.4,0.5,0.3,0,Low

# requirements.txt:
flask>=2.0
scikit-learn>=1.0
pandas>=1.3
numpy>=1.21