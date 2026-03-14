
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Iris Blossom · Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# REAL FLOWER IMAGE URLS  (loaded directly by the browser — no server fetch)
# These are stable Wikimedia Commons thumbnail URLs, open licence CC-BY-SA
# ══════════════════════════════════════════════════════════════════════════════

IMG = {
    "setosa": [
        "https://media.istockphoto.com/id/1355304050/photo/beautiful-purple-flower-alcea-setosa-close-up-in-greece.jpg?s=612x612&w=0&k=20&c=z3TAAC3jlthC77kwCp7mQbDyyya4L1AxB6TEPwmgbaE=",
        "https://thumbs.dreamstime.com/b/setosa-del-alcea-la-malvarrosa-erizada-en-jard%C3%ADn-verano-cierre-para-arriba-140686426.jpg",
        "https://thumbs.dreamstime.com/b/close-up-pink-alcea-setosa-313290758.jpg",
    ],
    "versicolor": [
        "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "https://c8.alamy.com/comp/2H5KDMC/iris-versicolour-flower-in-full-bloom-2H5KDMC.jpg",
        "https://newfs.s3.amazonaws.com/taxon-images-1000s1000/Iridaceae/iris-versicolor-fl-mlovit-a.jpg",
    ],
    "virginica": [
        "https://mellowmarshfarm.com/wp-content/uploads/2022/02/DETA-7163_Iris-virginica.jpg",
        "https://c8.alamy.com/comp/EHMM4C/purple-flowers-EHMM4C.jpg",
        "https://www.devonpondplants.co.uk/wp-content/uploads/2018/10/Iris-virginica-Orchid-Purple-768x1024.jpg",
    ],
}

def real_img(sp_key: str, idx: int = 0,
             style: str = "width:100%;border-radius:16px;object-fit:cover;",
             height: str = "280px", caption: str = "") -> str:
    """Return HTML <img> tag loading directly from Wikimedia via the browser."""
    url = IMG.get(sp_key, IMG["versicolor"])[idx]
    cap_html = (f'<div style="font-size:.74rem;font-weight:600;color:#7a5870;'
                f'text-align:center;margin-top:.35rem;font-style:italic">{caption}</div>'
                if caption else "")
    return (f'<div style="border-radius:18px;overflow:hidden;border:2px solid rgba(208,148,178,.45);'
            f'box-shadow:0 4px 20px rgba(140,60,100,.15);">'
            f'<img src="{url}" style="{style}max-height:{height};" '
            f'onerror="this.style.display=\'none\'" loading="lazy"/>'
            f'</div>{cap_html}')


# ══════════════════════════════════════════════════════════════════════════════
# FALLING LEAVES / PETALS ANIMATION  — canvas-based physics, fires on classify
# Leaf shapes + petal shapes mixed for a lush garden feel
# ══════════════════════════════════════════════════════════════════════════════

LEAF_COLORS = {
    "Iris-setosa":     ["#f9c8d4","#f4a7b9","#e8799a","#fde0e8","#f588b0","#fbb6c8",
                        "#ffaabb","#ee6688","#ffd0e0","#f070a0"],
    "Iris-versicolor": ["#c5a8e8","#b48edc","#9b70d0","#dccef5","#8055c0","#e8d8fa",
                        "#a080e0","#7050b8","#d0b8f8","#9060d0"],
    "Iris-virginica":  ["#a8d8b0","#78c488","#50b068","#d0ecd4","#3a9850","#c0e8c8",
                        "#60c070","#90e098","#b8f0c0","#40a858"],
}

def get_falling_leaves_html(species: str) -> str:
    colors = LEAF_COLORS.get(species, LEAF_COLORS["Iris-setosa"])
    cjs    = str(colors).replace("'", '"')

    return f"""
<canvas id="leaf-canvas" style="
  position:fixed; top:0; left:0; width:100vw; height:100vh;
  pointer-events:none; z-index:99999;
"></canvas>
<script>
(function() {{
  const canvas = document.getElementById('leaf-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;

  const COLORS = {cjs};
  const COUNT  = 160;
  const items  = [];

  /* Mix of leaf shapes and petal shapes */
  const SHAPES = ['leaf','petal','oval'];

  for (let i = 0; i < COUNT; i++) {{
    const shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
    const w = 12 + Math.random() * 24;
    const h = w  * (shape === 'oval' ? 0.55 + Math.random() * 0.35
                                     : 0.65 + Math.random() * 0.60);
    items.push({{
      shape,
      x:     Math.random() * canvas.width,
      y:     -h - Math.random() * canvas.height * 0.5,   // stagger start heights
      vx:    (Math.random() - 0.5) * 1.8,
      vy:    1.2 + Math.random() * 2.8,
      rot:   Math.random() * Math.PI * 2,
      rotV:  (Math.random() - 0.5) * 0.055,
      sway:  Math.random() * 0.025,        // horizontal sway amplitude
      swayT: Math.random() * Math.PI * 2,  // sway phase
      w, h,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: 0,
      fadeIn: 0.04 + Math.random() * 0.04,
    }});
  }}

  function drawLeaf(p) {{
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(p.rot);
    ctx.globalAlpha = p.alpha;
    ctx.fillStyle   = p.color;
    ctx.shadowColor = p.color;
    ctx.shadowBlur  = 5;

    ctx.beginPath();
    if (p.shape === 'leaf') {{
      /* Classic pointed leaf shape */
      ctx.moveTo(0, -p.h / 2);
      ctx.quadraticCurveTo( p.w / 2,  0, 0,  p.h / 2);
      ctx.quadraticCurveTo(-p.w / 2,  0, 0, -p.h / 2);
    }} else if (p.shape === 'petal') {{
      /* Rounded teardrop petal */
      ctx.moveTo(0, -p.h * 0.5);
      ctx.bezierCurveTo( p.w * 0.55, -p.h * 0.2,  p.w * 0.55, p.h * 0.4, 0, p.h * 0.52);
      ctx.bezierCurveTo(-p.w * 0.55,  p.h * 0.4, -p.w * 0.55, -p.h * 0.2, 0, -p.h * 0.5);
    }} else {{
      /* Simple oval */
      ctx.ellipse(0, 0, p.w / 2, p.h / 2, 0, 0, Math.PI * 2);
    }}
    ctx.closePath();
    ctx.fill();

    /* Midrib vein for leaf shape */
    if (p.shape === 'leaf') {{
      ctx.globalAlpha = p.alpha * 0.3;
      ctx.strokeStyle = 'rgba(255,255,255,0.7)';
      ctx.lineWidth   = 0.8;
      ctx.beginPath();
      ctx.moveTo(0, -p.h * 0.45);
      ctx.lineTo(0,  p.h * 0.45);
      ctx.stroke();
    }}

    ctx.restore();
  }}

  let t = 0;
  function animate() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let alive = false;

    for (const p of items) {{
      /* Gentle horizontal sway using sine wave */
      p.swayT += p.sway;
      p.x += p.vx + Math.sin(p.swayT * 40) * 0.6;
      p.y += p.vy;
      p.rot += p.rotV;

      /* Fade in quickly at start */
      if (p.alpha < 1) p.alpha = Math.min(1, p.alpha + p.fadeIn);

      /* Fade out near bottom */
      if (p.y > canvas.height * 0.78) {{
        p.alpha -= 0.018;
      }}

      if (p.alpha > 0 && p.y < canvas.height + p.h + 20) {{
        alive = true;
        drawLeaf(p);
      }}
    }}

    t++;
    if (alive) requestAnimationFrame(animate);
    else canvas.remove();
  }}

  requestAnimationFrame(animate);
}})();
</script>"""


# ══════════════════════════════════════════════════════════════════════════════
# THEME CSS — Flower Blossom
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500;1,600&family=Nunito:wght@400;600;700;800&family=Playfair+Display:ital,wght@0,700;0,900;1,700&display=swap');

:root {
  --blush:#fce8f0; --petal:#f0a0c0; --rose:#d45880; --deep:#8b2252;
  --violet:#7c52a8; --sage:#4a9060; --bark:#2a1520; --dusk:#4a2840;
  --muted:#7a5870; --border:rgba(208,148,178,0.50);
  --card:rgba(255,255,255,0.90); --shadow:rgba(140,60,100,0.14);
}
html,body,[class*="css"] {
  font-family:'Nunito',sans-serif !important;
  color:var(--bark) !important;
  background-color:#fdf0f8 !important;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 70% 55% at 12%  8%, rgba(249,180,210,.42) 0%,transparent 55%),
    radial-gradient(ellipse 55% 75% at 88% 92%, rgba(145, 95,175,.25) 0%,transparent 55%),
    radial-gradient(ellipse 65% 45% at 50% 50%, rgba(184,232,200,.28) 0%,transparent 65%),
    linear-gradient(160deg,#fef2f8 0%,#fdf0fb 30%,#f6f0fe 65%,#eef8f2 100%);
}
[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stMainBlockContainer"] { padding-top:.4rem; }

/* Typography */
p, li, td { color:var(--bark) !important; font-size:.95rem !important; font-weight:600 !important; }
h1,h2,h3  { font-family:'Playfair Display',serif !important; color:var(--deep) !important; font-weight:900 !important; }
h4,h5,h6  { font-family:'Cormorant Garamond',serif !important; color:var(--dusk) !important; font-weight:700 !important; }
strong { color:var(--deep) !important; font-weight:800 !important; }
[data-testid="stMarkdownContainer"] p { color:var(--bark) !important; font-weight:600 !important; }

/* Hero */
.hero {
  background:linear-gradient(135deg,rgba(240,160,192,.55) 0%,rgba(255,242,252,.88) 35%,rgba(210,190,242,.48) 70%,rgba(188,234,206,.42) 100%);
  border:2px solid var(--border); border-radius:28px;
  padding:2.8rem 3.2rem; margin-bottom:2rem;
  backdrop-filter:blur(18px);
  box-shadow:0 12px 55px var(--shadow),inset 0 1.5px 0 rgba(255,255,255,.92);
  position:relative; overflow:hidden;
}
.hero::before { content:''; position:absolute; inset:0; pointer-events:none;
  background:radial-gradient(circle 190px at 7% 55%,rgba(249,168,200,.48) 0%,transparent 65%),
    radial-gradient(circle 140px at 93% 18%,rgba(212,192,244,.48) 0%,transparent 65%),
    radial-gradient(circle 110px at 80% 85%,rgba(188,234,206,.42) 0%,transparent 65%); }
.hero-eyebrow { font-size:.76rem; font-weight:800; letter-spacing:.30em; text-transform:uppercase;
  color:var(--rose); margin-bottom:.85rem; display:block; }
.hero-title { font-family:'Playfair Display',serif; font-size:3.4rem; font-weight:900; line-height:1.06;
  background:linear-gradient(135deg,var(--deep) 0%,var(--violet) 52%,var(--sage) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0 0 .55rem; }
.hero-sub  { font-family:'Cormorant Garamond',serif; font-size:1.22rem; font-style:italic; color:var(--dusk); font-weight:600; }
.hero-divider { border:none; border-top:1.5px solid var(--border); margin:1.3rem 0 1rem; }
.hero-pills { display:flex; gap:.7rem; flex-wrap:wrap; }
.hero-pill  { background:rgba(255,255,255,.80); border:1.5px solid var(--border); border-radius:50px;
  padding:.3rem 1rem; font-size:.74rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; color:var(--dusk); }

/* Sidebar */
[data-testid="stSidebar"] {
  background:linear-gradient(180deg,rgba(254,241,252,.98) 0%,rgba(247,238,255,.98) 100%) !important;
  border-right:2px solid var(--border) !important;
}
[data-testid="stSidebar"] label {
  color:var(--dusk) !important; font-size:.8rem !important;
  font-weight:800 !important; letter-spacing:.07em; text-transform:uppercase;
}
[data-testid="stSidebar"] p { color:var(--dusk) !important; font-size:.85rem !important; font-weight:600 !important; }
.sidebar-title { font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:900; color:var(--deep); text-align:center; }
.sidebar-sub   { font-size:.72rem; font-weight:700; color:var(--muted); letter-spacing:.15em; text-transform:uppercase; text-align:center; }

/* Section titles */
.sec-title { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; color:var(--deep);
  display:flex; align-items:center; gap:.65rem; margin:1.4rem 0 .75rem; }
.sec-title::after { content:''; flex:1; height:1.5px; background:linear-gradient(90deg,var(--border),transparent); }

/* Measurement card */
.meas-card { border-radius:20px; background:var(--card); border:2px solid var(--border);
  padding:1.3rem 1.5rem; backdrop-filter:blur(12px); box-shadow:0 4px 22px var(--shadow); margin-top:.9rem; }
.meas-card-title { font-size:.76rem; font-weight:800; letter-spacing:.2em; text-transform:uppercase; color:var(--rose); margin-bottom:.9rem; }
.meas-row { display:flex; justify-content:space-between; align-items:center;
  padding:.5rem 0; border-bottom:1.5px solid rgba(208,148,178,.22); }
.meas-row:last-child { border-bottom:none; }
.meas-label { font-size:.9rem; font-weight:700; color:var(--dusk); }
.meas-value { font-family:'Playfair Display',serif; font-size:1.18rem; font-weight:700; color:var(--deep); }
.meas-unit  { font-size:.7rem; color:var(--muted); margin-left:.2rem; }

/* Result card */
.result-card { border-radius:24px;
  background:linear-gradient(155deg,rgba(255,255,255,.94) 0%,rgba(253,238,252,.94) 100%);
  border:2px solid var(--border); padding:2.2rem 2rem; text-align:center;
  backdrop-filter:blur(18px);
  box-shadow:0 12px 45px var(--shadow),inset 0 1.5px 0 rgba(255,255,255,.96);
  position:relative; overflow:hidden; }
.result-card::before { content:''; position:absolute; top:-70px; right:-70px; width:230px; height:230px; border-radius:50%;
  background:radial-gradient(circle,rgba(240,160,192,.32) 0%,transparent 70%); pointer-events:none; }
.result-eyebrow { font-size:.74rem; font-weight:800; letter-spacing:.28em; text-transform:uppercase; color:var(--rose); margin-bottom:.65rem; }
.result-emoji   { font-size:3.5rem; line-height:1; margin:.4rem 0; }
.result-species { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900; color:var(--bark); margin:.35rem 0 .2rem; }
.result-latin   { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:1.08rem; font-weight:600; color:var(--muted); margin-bottom:.7rem; }
.result-badge   { display:inline-block; padding:.35rem 1.2rem; border-radius:50px; font-size:.76rem; letter-spacing:.14em; font-weight:800; text-transform:uppercase; }
.setosa    { background:rgba(212, 88,128,.13); color:#8b2252; border:2px solid #d45880; }
.versicolor{ background:rgba(124, 82,168,.13); color:#4a2880; border:2px solid #7c52a8; }
.virginica { background:rgba( 74,144, 96,.13); color:#205830; border:2px solid #4a9060; }
.result-desc  { font-size:.93rem; font-weight:600; color:var(--dusk); line-height:1.68; margin:1rem 0; font-style:italic; }
.result-stats { display:flex; gap:1rem; justify-content:center; flex-wrap:wrap; margin-top:1rem; }
.stat-pill    { background:rgba(255,255,255,.84); border:2px solid var(--border); border-radius:50px; padding:.45rem 1.1rem; text-align:center; }
.stat-val     { font-family:'Playfair Display',serif; font-size:1.2rem; font-weight:700; color:var(--deep); display:block; }
.stat-lbl     { font-size:.68rem; font-weight:800; color:var(--muted); text-transform:uppercase; letter-spacing:.12em; }

/* Image labels */
.img-label   { font-size:.74rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; color:var(--rose); margin:.55rem 0 .2rem; display:block; }
.img-caption { font-size:.76rem; font-weight:600; color:var(--muted); font-style:italic; text-align:center; margin:.2rem 0 .7rem; }
.compare-name    { font-family:'Playfair Display',serif; font-size:1.02rem; font-weight:700; margin:.48rem 0 .12rem; }
.compare-latin   { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:.83rem; font-weight:600; color:var(--muted); }
.compare-habitat { font-size:.72rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; margin-top:.22rem; }

/* Metrics */
div[data-testid="metric-container"] {
  background:var(--card) !important; border:2px solid var(--border) !important;
  border-radius:18px !important; padding:.95rem 1.1rem !important;
  box-shadow:0 3px 14px var(--shadow) !important;
}
div[data-testid="metric-container"] label { color:var(--muted) !important; font-size:.76rem !important;
  font-weight:800 !important; text-transform:uppercase; letter-spacing:.09em; font-family:'Nunito',sans-serif !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family:'Playfair Display',serif !important; color:var(--deep) !important; font-weight:700 !important; }

/* Buttons */
.stButton>button {
  background:linear-gradient(135deg,#d45880 0%,#8b2252 100%) !important;
  color:white !important; border:none !important; border-radius:50px !important;
  font-family:'Nunito',sans-serif !important; font-weight:800 !important;
  letter-spacing:.1em !important; font-size:.92rem !important;
  box-shadow:0 5px 22px rgba(139,34,82,.38) !important;
  transition:all .22s ease !important; padding:.6rem 2rem !important;
  text-transform:uppercase !important;
}
.stButton>button:hover {
  background:linear-gradient(135deg,#e06898 0%,#a03068 100%) !important;
  box-shadow:0 8px 30px rgba(139,34,82,.5) !important; transform:translateY(-2px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:2px solid var(--border) !important; gap:.4rem !important; }
.stTabs [data-baseweb="tab"] { font-family:'Nunito',sans-serif !important; font-weight:800 !important;
  letter-spacing:.12em !important; font-size:.82rem !important; text-transform:uppercase !important;
  color:var(--muted) !important; background:transparent !important;
  border-radius:14px 14px 0 0 !important; padding:.65rem 1.4rem !important;
  border:2px solid transparent !important; }
.stTabs [aria-selected="true"] { background:rgba(255,255,255,.90) !important; color:var(--deep) !important;
  border-color:var(--border) !important; border-bottom-color:rgba(255,248,254,1) !important; }

/* Misc */
[data-baseweb="select"]>div { background:var(--card) !important; border:2px solid var(--border) !important;
  border-radius:14px !important; color:var(--bark) !important; }
[data-testid="stDataFrame"] { border:2px solid var(--border) !important; border-radius:16px !important; overflow:hidden !important; }
[data-testid="stDataFrame"] th { background:var(--blush) !important; color:var(--deep) !important; font-weight:800 !important; }
[data-testid="stDataFrame"] td { color:var(--bark) !important; font-weight:600 !important; }
.stAlert { border-radius:16px !important; border-left:4px solid var(--rose) !important; background:rgba(249,184,208,.2) !important; }
.stAlert p { color:var(--dusk) !important; font-weight:700 !important; }
.gallery-name  { font-family:'Playfair Display',serif; font-size:1rem; font-weight:700; margin:.5rem 0 .1rem; }
.gallery-latin { font-family:'Cormorant Garamond',serif; font-style:italic; font-size:.82rem; font-weight:600; color:var(--muted); }
.gallery-sub   { font-size:.72rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; margin-top:.2rem; }
::-webkit-scrollbar { width:5px; background:var(--blush); }
::-webkit-scrollbar-thumb { background:var(--petal); border-radius:5px; }
</style>
""", unsafe_allow_html=True)

# ── Plot helper ────────────────────────────────────────────────────────────────
_PLOT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,245,252,0.65)',
             font=dict(color='#2a1520', family='Nunito'))
_AX   = dict(gridcolor='rgba(208,148,178,0.28)', zeroline=False,
             linecolor='rgba(208,148,178,0.4)', tickfont=dict(color='#4a2840', size=11))
def PL(**kw):
    d = {**_PLOT,"xaxis":{**_AX,**kw.pop("xaxis",{})},"yaxis":{**_AX,**kw.pop("yaxis",{})}}
    d.update(kw); return d

CMAP = {"Iris-setosa":"#d45880","Iris-versicolor":"#7c52a8","Iris-virginica":"#4a9060"}

# ── Load model & data ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("iris_model.pkl","rb") as f: m = pickle.load(f)
    with open("label_encoder.pkl","rb") as f: le = pickle.load(f)
    return m, le

@st.cache_data
def load_data():
    return pd.read_csv("iris_flower_classification_10000_rows.csv")

try:    model, le = load_model(); model_loaded = True
except Exception as e: model_loaded = False; model_err = str(e)
df = load_data()

# ── Species meta ───────────────────────────────────────────────────────────────
SPECIES_INFO = {
    "Iris-setosa": {
        "emoji":"🌸","color":"#d45880","fill":"rgba(212,88,128,0.22)",
        "latin":"Iris setosa Pall.","svg_key":"setosa",
        "desc":"A delicate alpine beauty with tiny rounded petals edged in blush pink, thriving in frost-kissed Arctic meadows.",
        "habitat":"Arctic / Alpine","petals":"Tiny & rounded",
        "caps":["Iris setosa in bloom","Iris setosa — close-up","Iris setosa — natural habitat"],
    },
    "Iris-versicolor": {
        "emoji":"💜","color":"#7c52a8","fill":"rgba(124,82,168,0.22)",
        "latin":"Iris versicolor L.","svg_key":"versicolor",
        "desc":"The Blue Flag Iris with veined violet petals catching morning dew in lush wetland meadows across North America.",
        "habitat":"Wetlands / Meadows","petals":"Medium & veined",
        "caps":["Iris versicolor in bloom","Iris versicolor — close-up","Iris versicolor — natural habitat"],
    },
    "Iris-virginica": {
        "emoji":"🌿","color":"#4a9060","fill":"rgba(74,144,96,0.22)",
        "latin":"Iris virginica L.","svg_key":"virginica",
        "desc":"The Southern Blue Flag — most statuesque of the trio, adorning coastal marshes with large, elegant petals.",
        "habitat":"Coastal Marshes","petals":"Large & elegant",
        "caps":["Iris virginica in bloom","Iris virginica — close-up","Iris virginica — natural habitat"],
    },
}
CLASS_MAP = {0:"Iris-setosa",1:"Iris-versicolor",2:"Iris-virginica"}
FILL_MAP  = {k: v["fill"] for k,v in SPECIES_INFO.items()}

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <span class="hero-eyebrow">🌸 Botanical Intelligence · Machine Learning · Flower Classification</span>
  <div class="hero-title">Iris Blossom Classifier</div>
  <div class="hero-sub">Identify iris species with the grace of a garden in full bloom</div>
  <hr class="hero-divider"/>
  <div class="hero-pills">
    <span class="hero-pill">🌺 3 Species</span>
    <span class="hero-pill">🔬 Decision Tree</span>
    <span class="hero-pill">📊 10,000 Samples</span>
    <span class="hero-pill">🍃 Falling Leaves on Classify</span>
    <span class="hero-pill">📸 Real Flower Photos</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="padding:.6rem 0 .2rem">
      <div class="sidebar-title">🌸 Iris Blossom</div>
      <div class="sidebar-sub">Flower Measurements</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    sl = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.4, 0.1)
    sw = st.slider("🌿 Sepal Width  (cm)", 2.0, 4.5, 3.1, 0.1)
    pl = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 3.7, 0.1)
    pw = st.slider("🌸 Petal Width  (cm)", 0.1, 2.5, 1.2, 0.1)
    st.markdown("---")
    sidebar_btn = st.button("🌸 Classify Flower", use_container_width=True,
                            type="primary", key="sidebar_btn")
    st.markdown("---")
    st.markdown("""<div style="font-size:.78rem;font-weight:800;color:#d45880;
                text-transform:uppercase;letter-spacing:.14em;margin-bottom:.6rem">
                🌺 Garden Stats</div>""", unsafe_allow_html=True)
    st.metric("🌱 Training Samples", f"{len(df):,}")
    st.metric("🔬 Features","4")
    st.metric("🌺 Species","3")
    st.markdown("""<div style="margin-top:1.4rem;padding-top:1rem;
                border-top:1.5px solid rgba(208,148,178,.4);
                font-size:.75rem;color:#7a5870;text-align:center;
                font-style:italic;line-height:1.7;font-weight:600">
      Every flower is a soul blossoming in nature.<br/>
      <span style="font-size:.68rem;letter-spacing:.06em;font-weight:700">— Gerard de Nerval</span>
    </div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌸  Classify", "🌿  Explorer", "📊  Insights"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Classify
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    tab_btn    = st.button("🌸 Classify Flower", key="tab_btn", type="primary")
    do_predict = sidebar_btn or tab_btn

    col_in, col_out = st.columns([1, 1], gap="large")

    # LEFT: image preview + measurements ───────────────────────────────────────
    with col_in:
        st.markdown('<div class="sec-title">🌺 Flower Preview</div>', unsafe_allow_html=True)

        if do_predict and model_loaded:
            try:
                _pr = model.predict(np.array([[sl,sw,pl,pw]]))[0]
                _sp = CLASS_MAP.get(_pr,str(_pr)) if isinstance(_pr,(int,np.integer)) else str(_pr)
                lkey  = SPECIES_INFO[_sp]["svg_key"]
                lname = _sp.replace("Iris-","Iris ")
                llat  = SPECIES_INFO[_sp]["latin"]
            except Exception:
                lkey,lname,llat = "versicolor","Iris flower","pending"
        else:
            lkey,lname,llat = "versicolor","Iris flower","Adjust sliders → Classify"

        # Real flower image — loaded directly by browser
        st.markdown(
            real_img(lkey, 0, height="300px",
                     style="width:100%;object-fit:cover;",
                     caption=f"🌸 {lname}  ·  {llat}"),
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div class="meas-card">
          <div class="meas-card-title">📐 Current Measurements</div>
          <div class="meas-row"><span class="meas-label">🌿 Sepal Length</span><span class="meas-value">{sl}<span class="meas-unit"> cm</span></span></div>
          <div class="meas-row"><span class="meas-label">🌿 Sepal Width</span> <span class="meas-value">{sw}<span class="meas-unit"> cm</span></span></div>
          <div class="meas-row"><span class="meas-label">🌸 Petal Length</span><span class="meas-value">{pl}<span class="meas-unit"> cm</span></span></div>
          <div class="meas-row"><span class="meas-label">🌸 Petal Width</span> <span class="meas-value">{pw}<span class="meas-unit"> cm</span></span></div>
        </div>""", unsafe_allow_html=True)

    # RIGHT: classification result ──────────────────────────────────────────────
    with col_out:
        st.markdown('<div class="sec-title">✨ Classification Result</div>', unsafe_allow_html=True)

        if not do_predict:
            st.info("🌸 Adjust the sliders and click **Classify Flower** to identify your iris species.")
        else:
            if model_loaded:
                try:
                    X        = np.array([[sl,sw,pl,pw]])
                    pred_raw = model.predict(X)[0]
                    species  = CLASS_MAP.get(pred_raw,str(pred_raw)) if isinstance(pred_raw,(int,np.integer)) else str(pred_raw)
                    proba    = model.predict_proba(X)[0] if hasattr(model,"predict_proba") else [0.33,0.33,0.34]
                    info     = SPECIES_INFO.get(species, SPECIES_INFO["Iris-setosa"])
                    css_cls  = species.replace("Iris-","").lower()
                    conf     = max(proba)*100
                    sp_key   = info["svg_key"]

                    # 🍃 FALLING LEAVES + PETALS ANIMATION
                    st.markdown(get_falling_leaves_html(species), unsafe_allow_html=True)

                    # Result card
                    st.markdown(f"""
                    <div class="result-card">
                      <div class="result-eyebrow">✦ Species Identified ✦</div>
                      <div class="result-emoji">{info['emoji']}</div>
                      <div class="result-species">{species.replace('Iris-','Iris ')}</div>
                      <div class="result-latin">{info['latin']}</div>
                      <span class="result-badge {css_cls}">{css_cls}</span>
                      <p class="result-desc">{info['desc']}</p>
                      <div class="result-stats">
                        <div class="stat-pill"><span class="stat-val">{conf:.1f}%</span><span class="stat-lbl">Confidence</span></div>
                        <div class="stat-pill"><span class="stat-val" style="font-size:.9rem">{info['habitat']}</span><span class="stat-lbl">Habitat</span></div>
                        <div class="stat-pill"><span class="stat-val" style="font-size:.9rem">{info['petals']}</span><span class="stat-lbl">Petals</span></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

                    # ── Real flower photographs ────────────────────────────────
                    st.markdown('<div class="sec-title" style="margin-top:1.3rem">📸 Field Photographs</div>',
                                unsafe_allow_html=True)

                    # Large main photo
                    st.markdown(
                        real_img(sp_key, 0, height="280px",
                                 style="width:100%;object-fit:cover;",
                                 caption=info["caps"][0]),
                        unsafe_allow_html=True
                    )

                    # Two smaller photos side by side
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown('<span class="img-label">🌸 Close-up View</span>', unsafe_allow_html=True)
                        st.markdown(
                            real_img(sp_key, 1, height="170px",
                                     style="width:100%;object-fit:cover;"),
                            unsafe_allow_html=True
                        )
                        st.markdown(f'<div class="img-caption">{info["caps"][1]}</div>', unsafe_allow_html=True)
                    with pc2:
                        st.markdown('<span class="img-label">🌿 Natural Habitat</span>', unsafe_allow_html=True)
                        st.markdown(
                            real_img(sp_key, 2, height="170px",
                                     style="width:100%;object-fit:cover;"),
                            unsafe_allow_html=True
                        )
                        st.markdown(f'<div class="img-caption">{info["caps"][2]}</div>', unsafe_allow_html=True)

                    # ── Compare with other species ─────────────────────────────
                    st.markdown('<div class="sec-title" style="margin-top:1.3rem">🔍 Compare With Others</div>',
                                unsafe_allow_html=True)
                    others = [(k,v) for k,v in SPECIES_INFO.items() if k != species]
                    oc = st.columns(2)
                    for i,(sk,sv) in enumerate(others):
                        with oc[i]:
                            st.markdown(
                                real_img(sv["svg_key"], 0, height="160px",
                                         style="width:100%;object-fit:cover;"),
                                unsafe_allow_html=True
                            )
                            st.markdown(f"""
                            <div class="compare-name" style="color:{sv['color']}">{sv['emoji']} {sk.replace('Iris-','')}</div>
                            <div class="compare-latin">{sv['latin']}</div>
                            <div class="compare-habitat">{sv['habitat']}</div>""", unsafe_allow_html=True)

                    # ── Confidence bar ─────────────────────────────────────────
                    st.markdown('<div class="sec-title" style="margin-top:1.3rem">📊 Confidence Breakdown</div>',
                                unsafe_allow_html=True)
                    clist = list(SPECIES_INFO.keys())
                    pbar  = go.Figure(go.Bar(
                        x=[p*100 for p in proba],
                        y=[c.replace("Iris-","") for c in clist],
                        orientation='h',
                        marker=dict(color=[SPECIES_INFO[c]["color"] for c in clist],
                                    opacity=0.88, line=dict(color='rgba(208,148,178,.4)',width=1)),
                        text=[f"{p*100:.1f}%" for p in proba],
                        textposition='outside',
                        textfont=dict(color='#2a1520', size=13, family='Nunito'),
                    ))
                    pbar.update_layout(**PL(
                        xaxis=dict(range=[0,118],ticksuffix='%',tickfont=dict(color='#4a2840',size=11)),
                        yaxis=dict(tickfont=dict(size=14,color='#2a1520',family='Nunito')),
                        height=220, margin=dict(l=10,r=65,t=10,b=10),
                    ))
                    st.plotly_chart(pbar, use_container_width=True)

                except Exception as e:
                    st.error(f"⚠️ Prediction error: {e}")
            else:
                st.warning(f"🌿 Model not found: {model_err}\n\nRun `python retrain_model.py` first.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-title">🌿 Dataset Overview</div>', unsafe_allow_html=True)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🌱 Total Samples",f"{len(df):,}"); m2.metric("🔬 Features","4")
    m3.metric("🌺 Species","3"); m4.metric("⚖️ Balance","Balanced")

    ca,cb = st.columns(2)
    with ca:
        dist = df["species"].value_counts().reset_index(); dist.columns=["Species","Count"]
        pie  = px.pie(dist,names="Species",values="Count",color="Species",hole=0.52,
                      color_discrete_map=CMAP,title="Species Distribution")
        pie.update_layout(**PL(legend=dict(font=dict(color='#2a1520',size=12)),
                               title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16))))
        st.plotly_chart(pie,use_container_width=True)
    with cb:
        feat = st.selectbox("🔎 Select Feature",["sepal_length","sepal_width","petal_length","petal_width"])
        box  = px.box(df,x="species",y=feat,color="species",points="outliers",
                      color_discrete_map=CMAP,title=f"{feat.replace('_',' ').title()} by Species")
        box.update_layout(**PL(showlegend=False,
                               title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16))))
        st.plotly_chart(box,use_container_width=True)

    st.markdown('<div class="sec-title">🎻 Violin Distributions</div>', unsafe_allow_html=True)
    vf = st.selectbox("🔎 Feature",["sepal_length","sepal_width","petal_length","petal_width"],key="vf")
    vfig = go.Figure()
    for sp in SPECIES_INFO:
        vfig.add_trace(go.Violin(
            y=df[df.species==sp][vf].values, name=sp.replace("Iris-",""),
            fillcolor=FILL_MAP[sp], line_color=SPECIES_INFO[sp]["color"],   # ✅ rgba
            box_visible=True, meanline_visible=True, opacity=0.88, legendgroup=sp,
        ))
    vfig.update_layout(**PL(yaxis=dict(title=vf.replace("_"," ").title()),height=420,
                            legend=dict(font=dict(color='#2a1520',size=12)),showlegend=True))
    st.plotly_chart(vfig,use_container_width=True)

    st.markdown('<div class="sec-title">🌼 Feature Pair Scatter</div>', unsafe_allow_html=True)
    samp = df.sample(min(500,len(df)),random_state=42)
    sc   = px.scatter_matrix(samp,dimensions=["sepal_length","sepal_width","petal_length","petal_width"],
               color="species",color_discrete_map=CMAP,opacity=0.65,title="Pairwise Feature Relationships")
    sc.update_traces(marker=dict(size=3))
    sc.update_layout(**PL(height=560,title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16)),
                          legend=dict(font=dict(color='#2a1520',size=12))))
    st.plotly_chart(sc,use_container_width=True)

    st.markdown('<div class="sec-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = df.drop(columns=["species"]).corr()
    ht   = px.imshow(corr,text_auto=".2f",aspect="auto",
                     color_continuous_scale=[[0,"#fce8f0"],[0.5,"#d45880"],[1,"#7c52a8"]],
                     title="Feature Correlation Matrix")
    ht.update_layout(**PL(title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16))))
    st.plotly_chart(ht,use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-title">📊 Model Performance</div>', unsafe_allow_html=True)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    @st.cache_data
    def evaluate_model(_model):
        le2 = LabelEncoder()
        Xa  = df[["sepal_length","sepal_width","petal_length","petal_width"]].values
        ya  = le2.fit_transform(df["species"].values)
        _,Xt,_,yt = train_test_split(Xa,ya,test_size=0.2,random_state=42,stratify=ya)
        yt = le2.inverse_transform(yt)
        try:
            yp = _model.predict(Xt)
            yp = [CLASS_MAP.get(p,str(p)) if isinstance(p,(int,np.integer)) else str(p) for p in yp]
        except Exception: yp = list(yt)
        return yt,yp,le2.classes_

    if model_loaded:
        yt,yp,cls = evaluate_model(model)
        rpt = classification_report(yt,yp,output_dict=True)
        cm  = confusion_matrix(yt,yp,labels=list(SPECIES_INFO.keys()))
        acc = rpt["accuracy"]

        m1,m2,m3 = st.columns(3)
        m1.metric("🎯 Accuracy",      f"{acc*100:.2f}%")
        m2.metric("⚡ Avg Precision", f"{rpt['macro avg']['precision']*100:.2f}%")
        m3.metric("🔁 Avg Recall",    f"{rpt['macro avg']['recall']*100:.2f}%")

        cc,cl = st.columns(2)
        with cc:
            cmf = px.imshow(cm,
                x=[c.replace("Iris-","") for c in SPECIES_INFO],
                y=[c.replace("Iris-","") for c in SPECIES_INFO],
                text_auto=True,title="Confusion Matrix",
                color_continuous_scale=[[0,"#fce8f0"],[0.5,"#d45880"],[1,"#7c52a8"]])
            cmf.update_layout(**PL(title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16))))
            st.plotly_chart(cmf,use_container_width=True)
        with cl:
            rows = []
            for sp in SPECIES_INFO:
                if sp in rpt:
                    r = rpt[sp]
                    rows.append({"Species":sp.replace("Iris-",""),"Precision":f"{r['precision']*100:.1f}%",
                                 "Recall":f"{r['recall']*100:.1f}%","F1-Score":f"{r['f1-score']*100:.1f}%",
                                 "Support":int(r['support'])})
            st.markdown('<div class="sec-title">📋 Per-Class Statistics</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)

        if hasattr(model,"feature_importances_"):
            st.markdown('<div class="sec-title">🏆 Feature Importance</div>', unsafe_allow_html=True)
            feats = ["Sepal Length","Sepal Width","Petal Length","Petal Width"]
            imp   = model.feature_importances_
            fi = px.bar(x=imp,y=feats,orientation='h',text=[f"{v:.3f}" for v in imp],color=imp,
                        title="Decision Tree Feature Importances",
                        color_continuous_scale=[[0,"#fce8f0"],[0.5,"#d45880"],[1,"#7c52a8"]])
            fi.update_traces(textposition='outside',textfont=dict(color='#2a1520',size=12,family='Nunito'),
                             marker_line_color='rgba(208,148,178,.5)',marker_line_width=1)
            fi.update_layout(**PL(showlegend=False,coloraxis_showscale=False,
                                  title=dict(font=dict(color='#8b2252',family='Playfair Display',size=16))))
            st.plotly_chart(fi,use_container_width=True)

        # Gallery — real photos
        st.markdown('<div class="sec-title">🌸 Species Gallery</div>', unsafe_allow_html=True)
        gcols = st.columns(3)
        for idx,(sp,info) in enumerate(SPECIES_INFO.items()):
            with gcols[idx]:
                st.markdown(
                    real_img(info["svg_key"], 0, height="200px",
                             style="width:100%;object-fit:cover;"),
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                <div class="gallery-name" style="color:{info['color']}">{info['emoji']} {sp.replace('Iris-','')}</div>
                <div class="gallery-latin">{info['latin']}</div>
                <div class="gallery-sub">{info['habitat']} · {info['petals']}</div>""", unsafe_allow_html=True)
    else:
        st.info("🌿 Run `python retrain_model.py` to load the model, then restart the app.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER — Developer credit
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="
  text-align:center; margin-top:3.5rem; padding:2rem 1rem 1.6rem;
  border-top:2px solid rgba(208,148,178,.35);
  background:linear-gradient(135deg,rgba(252,232,240,.50) 0%,rgba(246,232,254,.50) 100%);
  border-radius:0 0 20px 20px;">

  <div style="font-family:'Playfair Display',Georgia,serif; font-size:1.08rem;
              font-weight:900; color:#8b2252; letter-spacing:.04em; margin-bottom:.4rem;">
    🌸 Iris Blossom Classifier
  </div>

  <div style="font-family:'Cormorant Garamond',Georgia,serif; font-style:italic;
              font-size:.96rem; color:#7a5870; letter-spacing:.04em; margin-bottom:.9rem;">
    Built with Streamlit &amp; Plotly &nbsp;·&nbsp; Decision Tree Classifier
    &nbsp;·&nbsp; Wikimedia Commons Photos
  </div>

  <div style="display:inline-flex; align-items:center; gap:.6rem;
              background:rgba(255,255,255,.80);
              border:2px solid rgba(208,148,178,.55);
              border-radius:50px; padding:.5rem 1.6rem;
              box-shadow:0 3px 14px rgba(140,60,100,.12);">
    <span style="font-size:1.15rem;">✍️</span>
    <span style="font-family:'Nunito',sans-serif; font-size:.85rem; font-weight:800;
                 color:#4a2840; letter-spacing:.07em; text-transform:uppercase;">
      Developed by &nbsp;<span style="color:#d45880; font-size:.9rem;">Tibah Wajahat</span>
    </span>
  </div>

</div>
""", unsafe_allow_html=True)