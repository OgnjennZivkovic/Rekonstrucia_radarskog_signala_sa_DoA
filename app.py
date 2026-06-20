"""Deployment UI for the radar antenna-reconstruction ensemble.

A small Flask web app that loads the trained ensemble (ensemble.joblib) and lets
a user:
  1. generate a random radar pulse (random number of targets at random angles,
     built from ULA steering vectors + noise, scaled to the training distribution),
  2. choose how many antennas to mask (hide),
  3. see the reconstructed signal plotted next to the true signal.

Run:
    pip install flask
    python app.py --ensemble result_e/checkpoints/ensemble.joblib
    # open http://127.0.0.1:5000

The app uses the SAME model the project trained -- this is the real deployment,
not a simulation. Normalization stats and array geometry are read from the saved
ensemble, so it adapts to whatever model you load.
"""

import argparse
import numpy as np
from flask import Flask, request, jsonify, render_template_string

from ensemble import load_stacker

app = Flask(__name__)
STATE = {"stack": None, "cfg": None, "stats": None}


def steering(angle_deg, M, d=0.5):
    m = np.arange(M)
    return np.exp(-1j * 2 * np.pi * d * m * np.sin(np.deg2rad(angle_deg)))


def generate_pulse(cfg, stats, snr_db, rng):
    """One random pulse: K targets at random angles, ULA steering + noise,
    scaled so each real/imag channel matches the training std."""
    M = cfg.M
    K = int(rng.integers(cfg.min_targets, cfg.max_targets + 1))
    angles = np.sort(rng.uniform(cfg.angle_min, cfg.angle_max, size=K))
    A = np.stack([steering(a, M) for a in angles], axis=1)            # (M, K)
    s = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
    clean = A @ s
    sp = np.mean(np.abs(clean) ** 2)
    npow = sp / (10 ** (snr_db / 10))
    noise = np.sqrt(npow / 2) * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
    sig = clean + noise
    target_std = float(np.mean(stats["std"]))
    cur = np.std(np.concatenate([sig.real, sig.imag]))
    if cur > 0:
        sig = sig / cur * target_std
    return sig.astype(np.complex64), [float(a) for a in angles], K


def reconstruct(cfg, stats, stack, sig, n_mask, rng):
    """Mask n_mask antennas, run the ensemble, return true + reconstructed."""
    M = cfg.M
    real = np.stack([sig.real, sig.imag], axis=-1).astype(np.float32)  # (M, 2)
    mean = stats["mean"].reshape(1, 2)
    std = stats["std"].reshape(1, 2)
    xn = (real - mean) / std

    masked_idx = np.sort(rng.choice(M, size=int(n_mask), replace=False))
    mask = np.ones(M, dtype=np.float32)
    mask[masked_idx] = 0.0
    masked = xn * mask[:, None]

    X = np.concatenate([masked.reshape(-1), mask])[None, :].astype(np.float32)  # (1, 3M)
    pred = stack.predict(X).reshape(M, 2)
    rec = pred * std + mean                                            # de-normalize

    e = (rec[masked_idx] - real[masked_idx])
    t = real[masked_idx]
    nmse = float((e ** 2).sum() / ((t ** 2).sum() + 1e-12))
    nmse_db = float(10 * np.log10(nmse + 1e-12))
    mae = float(np.sqrt((e ** 2).sum(axis=1)).mean())

    return {
        "true_re": real[:, 0].tolist(), "true_im": real[:, 1].tolist(),
        "rec_re": rec[:, 0].tolist(), "rec_im": rec[:, 1].tolist(),
        "masked": masked_idx.tolist(),
        "nmse_db": nmse_db, "mae": mae,
    }


@app.route("/")
def index():
    return render_template_string(PAGE, M=STATE["cfg"].M)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    cfg, stats = STATE["cfg"], STATE["stats"]
    snr = float(request.json.get("snr_db", 15.0))
    rng = np.random.default_rng()
    sig, angles, K = generate_pulse(cfg, stats, snr, rng)
    return jsonify({"re": sig.real.tolist(), "im": sig.imag.tolist(),
                    "angles": angles, "K": K, "M": cfg.M})


@app.route("/api/reconstruct", methods=["POST"])
def api_reconstruct():
    cfg, stats, stack = STATE["cfg"], STATE["stats"], STATE["stack"]
    data = request.json
    sig = np.array(data["re"], dtype=np.float64) + 1j * np.array(data["im"], dtype=np.float64)
    n_mask = int(data["n_mask"])
    n_mask = max(1, min(n_mask, cfg.M - 1))
    rng = np.random.default_rng()
    return jsonify(reconstruct(cfg, stats, stack, sig.astype(np.complex64), n_mask, rng))


PAGE = """
<!doctype html><html lang="sr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rekonstrukcija radarskog signala</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root{--bg:#0f1320;--card:#181d2e;--ink:#e8ecf6;--mut:#9aa6c0;--acc:#6f7be0;--good:#1d9e75;--mask:#d85a30;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:24px}
  h1{font-size:22px;margin:0 0 4px}
  p.sub{color:var(--mut);margin:0 0 20px}
  .card{background:var(--card);border:1px solid #242b42;border-radius:14px;padding:18px;margin-bottom:16px}
  .row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-end}
  label{display:block;font-size:13px;color:var(--mut);margin-bottom:6px}
  input[type=range]{width:220px;vertical-align:middle}
  .val{display:inline-block;min-width:34px;color:var(--ink);font-weight:600}
  button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:10px 16px;font-size:14px;font-weight:600;cursor:pointer}
  button.ghost{background:#222a42;color:var(--ink)}
  button:disabled{opacity:.5;cursor:not-allowed}
  .info{display:flex;gap:24px;flex-wrap:wrap;margin-top:10px;color:var(--mut);font-size:13px}
  .info b{color:var(--ink)}
  #plot{width:100%;height:560px}
  .pill{display:inline-block;background:#222a42;border-radius:999px;padding:2px 10px;margin:2px 4px 2px 0;font-size:13px}
</style></head>
<body><div class="wrap">
  <h1>Rekonstrukcija signala sa maskiranih antena</h1>
  <p class="sub">Generiši nasumičan radarski signal, izaberi koliko antena se maskira, pa pogledaj rekonstrukciju ansambla naspram stvarnog signala. Niz ima {{M}} antena.</p>

  <div class="card">
    <div class="row">
      <div>
        <label>SNR generisanog signala: <span class="val" id="snrv">15</span> dB</label>
        <input type="range" id="snr" min="-5" max="30" value="15" step="1">
      </div>
      <button id="gen">Generiši nasumičan signal</button>
    </div>
    <div class="info" id="siginfo"></div>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>Broj maskiranih antena: <span class="val" id="nmv">5</span></label>
        <input type="range" id="nm" min="1" max="{{M-1}}" value="5" step="1">
      </div>
      <button id="rec" class="ghost" disabled>Rekonstruiši</button>
    </div>
    <div class="info" id="recinfo"></div>
  </div>

  <div class="card"><div id="plot"></div></div>
</div>
<script>
let SIG=null, M={{M}};
const $=id=>document.getElementById(id);
$("snr").oninput=e=>$("snrv").textContent=e.target.value;
$("nm").oninput=e=>$("nmv").textContent=e.target.value;

function layout(){
  const ax={gridcolor:"#2a3252",zerolinecolor:"#2a3252",color:"#9aa6c0"};
  return {paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#e8ecf6"},
    margin:{t:30,r:10,b:45,l:50},legend:{orientation:"h",y:1.12},
    grid:{rows:2,columns:1,pattern:"independent",roworder:"top to bottom"},
    xaxis:{...ax},yaxis:{...ax,title:"Re"},
    xaxis2:{...ax,title:"indeks antene"},yaxis2:{...ax,title:"Im"}};
}
function drawTrue(){
  const x=[...Array(M).keys()];
  Plotly.newPlot("plot",[
   {x,y:SIG.re,name:"stvarni (Re)",mode:"lines+markers",line:{color:"#1d9e75"},xaxis:"x",yaxis:"y"},
   {x,y:SIG.im,name:"stvarni (Im)",mode:"lines+markers",line:{color:"#1d9e75"},xaxis:"x2",yaxis:"y2",showlegend:false},
  ],layout(),{displayModeBar:false});
}
function drawRecon(r){
  const x=[...Array(M).keys()];
  const mx=r.masked, mre=mx.map(i=>r.true_re[i]), mim=mx.map(i=>r.true_im[i]);
  Plotly.newPlot("plot",[
   {x,y:r.true_re,name:"stvarni (Re)",mode:"lines+markers",line:{color:"#1d9e75"},xaxis:"x",yaxis:"y"},
   {x,y:r.rec_re,name:"rekonstruisan (Re)",mode:"lines+markers",line:{color:"#6f7be0",dash:"dash"},xaxis:"x",yaxis:"y"},
   {x:mx,y:mre,name:"maskirana antena",mode:"markers",marker:{size:13,color:"rgba(0,0,0,0)",line:{color:"#d85a30",width:2}},xaxis:"x",yaxis:"y"},
   {x,y:r.true_im,name:"stvarni (Im)",mode:"lines+markers",line:{color:"#1d9e75"},xaxis:"x2",yaxis:"y2",showlegend:false},
   {x,y:r.rec_im,name:"rekonstruisan (Im)",mode:"lines+markers",line:{color:"#6f7be0",dash:"dash"},xaxis:"x2",yaxis:"y2",showlegend:false},
   {x:mx,y:mim,mode:"markers",marker:{size:13,color:"rgba(0,0,0,0)",line:{color:"#d85a30",width:2}},xaxis:"x2",yaxis:"y2",showlegend:false},
  ],layout(),{displayModeBar:false});
}

$("gen").onclick=async()=>{
  const r=await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({snr_db:+$("snr").value})});
  SIG=await r.json();
  const ang=SIG.angles.map(a=>`<span class="pill">${a.toFixed(1)}°</span>`).join("");
  $("siginfo").innerHTML=`<div><b>${SIG.K}</b> meta &nbsp; uglovi: ${ang}</div>`;
  $("rec").disabled=false; $("recinfo").innerHTML="";
  drawTrue();
};
$("rec").onclick=async()=>{
  if(!SIG)return;
  const r=await fetch("/api/reconstruct",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({re:SIG.re,im:SIG.im,n_mask:+$("nm").value})});
  const d=await r.json();
  $("recinfo").innerHTML=`<div>maskirano: <b>${d.masked.length}</b> antena</div>
     <div>NMSE na maskiranim: <b>${d.nmse_db.toFixed(2)} dB</b></div>
     <div>MAE: <b>${d.mae.toFixed(3)}</b></div>`;
  drawRecon(d);
};
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="result_e/checkpoints/ensemble.joblib")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    stack = load_stacker(args.ensemble)
    STATE["stack"] = stack
    STATE["cfg"] = stack.cfg
    STATE["stats"] = stack.stats
    print(f"Loaded ensemble: members={stack.cfg.members}, M={stack.cfg.M}")
    print(f"Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
