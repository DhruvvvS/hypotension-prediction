import json
import os
from http.server import BaseHTTPRequestHandler

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "..", "models")

booster = xgb.Booster()
booster.load_model(os.path.join(MODEL_DIR, "xgb.json"))
meta = json.load(open(os.path.join(MODEL_DIR, "meta.json")))
FEATURES = meta["features"]


def window_features(obs, age):
    # must match src/data_loader.py
    x = np.arange(len(obs))
    return {
        "cur_map": obs[-1], "mean_map": obs.mean(), "std_map": obs.std(),
        "min_map": obs.min(), "slope": np.polyfit(x, obs, 1)[0],
        "delta5": obs[-1] - obs[-6], "age": age,
    }


def predict(readings, age):
    obs = np.array(readings, dtype=float)
    feats = window_features(obs, age)
    row = np.array([[feats[f] for f in FEATURES]])
    dm = xgb.DMatrix(row, feature_names=FEATURES)
    risk = float(booster.predict(dm)[0])
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]  # last value is the bias
    top = sorted(zip(FEATURES, contribs), key=lambda t: -abs(t[1]))[:5]
    return {
        "risk": risk,
        "alert": risk >= meta["threshold"],
        "threshold": meta["threshold"],
        "drivers": [{"feature": f, "value": float(feats[f]), "push": float(c)} for f, c in top],
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>IOH early warning</title>
<style>
  :root { --bg:#f4f6f5; --card:#fff; --ink:#1d2b2a; --muted:#667573; --teal:#0f766e; --red:#b42318; --line:#dde4e2; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:640px; margin:0 auto; padding:32px 18px 60px; }
  h1 { font-size:1.5rem; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:20px; margin-bottom:16px; }
  label { display:block; font-weight:600; margin:0 0 6px; }
  input, textarea { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:8px; font:inherit; }
  input:focus, textarea:focus { outline:2px solid var(--teal); border-color:transparent; }
  .row { margin-bottom:14px; }
  small { color:var(--muted); }
  button { background:var(--teal); color:#fff; border:0; padding:11px 20px; border-radius:8px; font:inherit; font-weight:600; cursor:pointer; }
  button:hover { background:#0b5f58; }
  .big { font-size:2.4rem; font-weight:700; }
  .ok { color:var(--teal); } .bad { color:var(--red); }
  .bar { height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:4px 0 12px; }
  .bar span { display:block; height:100%; }
  .drv { display:flex; justify-content:space-between; font-size:.92rem; }
  .note { font-size:.82rem; color:var(--muted); }
</style></head>
<body><main>
  <h1>Intraoperative hypotension early warning</h1>
  <p class="sub">Risk of a sustained MAP drop below 65 mmHg in the next 15 minutes.</p>

  <div class="card">
    <div class="row">
      <label for="maps">Last 10 MAP readings (one per minute, oldest first)</label>
      <textarea id="maps" rows="2">84, 83, 82, 80, 79, 77, 76, 74, 73, 71</textarea>
      <small>Comma separated, mmHg.</small>
    </div>
    <div class="row">
      <label for="age">Age</label>
      <input id="age" type="number" value="62" min="1" max="110">
    </div>
    <button id="go">Estimate risk</button>
  </div>

  <div class="card" id="out" hidden>
    <div class="big" id="risk"></div>
    <div id="verdict"></div>
    <h3 style="margin:18px 0 8px">What moved the estimate</h3>
    <div id="drivers"></div>
  </div>

  <p class="note">Research prototype built from a course project. Not validated for clinical use.</p>
</main>
<script>
const $ = id => document.getElementById(id);
$("go").onclick = async () => {
  const readings = $("maps").value.split(",").map(Number).filter(n => !isNaN(n));
  if (readings.length !== 10) { alert("Please enter exactly 10 readings."); return; }
  const res = await fetch("/api/index", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({readings, age: Number($("age").value)})
  });
  const d = await res.json();
  $("out").hidden = false;
  $("risk").textContent = (d.risk * 100).toFixed(1) + "%";
  $("risk").className = "big " + (d.alert ? "bad" : "ok");
  $("verdict").textContent = d.alert
    ? "Above the alert threshold. Worth a closer look at the trend."
    : "Below the alert threshold.";
  const max = Math.max(...d.drivers.map(x => Math.abs(x.push)), 0.001);
  $("drivers").innerHTML = d.drivers.map(x => `
    <div class="drv"><span>${x.feature} (${x.value.toFixed(1)})</span>
    <span>${x.push > 0 ? "raises" : "lowers"} risk</span></div>
    <div class="bar"><span style="width:${(Math.abs(x.push)/max*100).toFixed(0)}%;background:${x.push > 0 ? "#b42318" : "#0f766e"}"></span></div>`).join("");
};
</script></body></html>"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        size = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(size))
            out, code = predict(payload["readings"], float(payload["age"])), 200
        except Exception as e:
            out, code = {"error": str(e)}, 400
        body = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)