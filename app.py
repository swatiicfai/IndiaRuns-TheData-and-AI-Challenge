import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(
    page_title="Redrob · Intelligent Candidate Discovery",
    layout="wide",
    page_icon="🔴",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Redrob.io Dark Theme
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,300;1,400;1,700;1,800;1,900&display=swap');

/* ── Reset & Base ──────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #080808 !important;
    color: #ffffff !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
.stApp { background: #080808 !important; }

/* ── TOP NAV ──────────────────────────────────────────────── */
.topnav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.2rem 3rem;
    background: rgba(8,8,8,0.95);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: sticky; top: 0; z-index: 100;
}
.topnav-logo {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 1.3rem; font-weight: 800; color: white; letter-spacing: -0.02em;
}
.topnav-logo .logo-icon {
    width: 28px; height: 28px; background: #E63946;
    border-radius: 6px; display: inline-flex; align-items: center;
    justify-content: center; font-size: 0.9rem; font-weight: 900; color: white;
}
.topnav-links { display: flex; gap: 2rem; }
.topnav-links a {
    color: rgba(255,255,255,0.65); font-size: 0.875rem;
    font-weight: 500; text-decoration: none; transition: color 0.2s;
}
.topnav-links a:hover { color: white; }
.topnav-cta {
    background: white; color: #080808 !important;
    border-radius: 8px; padding: 0.5rem 1.2rem;
    font-size: 0.875rem; font-weight: 700; text-decoration: none;
    transition: background 0.2s;
}

/* ── HERO ──────────────────────────────────────────────────── */
.hero-wrap {
    position: relative; overflow: hidden;
    min-height: 520px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center;
    padding: 6rem 2rem 5rem;
    background: #080808;
}
/* Colorful ray burst — pure CSS */
.hero-rays {
    position: absolute; inset: 0;
    background:
        conic-gradient(
            from 180deg at 50% 60%,
            rgba(230,57,70,0.0)   0deg,
            rgba(230,57,70,0.18)  15deg,
            rgba(255,120,50,0.12) 30deg,
            rgba(255,200,50,0.10) 50deg,
            rgba(100,200,120,0.08)70deg,
            rgba(50,150,255,0.12) 90deg,
            rgba(120,80,255,0.15) 110deg,
            rgba(200,50,200,0.10) 130deg,
            rgba(230,57,70,0.15)  150deg,
            rgba(255,80,80,0.08)  180deg,
            rgba(255,150,50,0.06) 210deg,
            rgba(50,200,200,0.08) 240deg,
            rgba(80,100,255,0.10) 270deg,
            rgba(150,50,230,0.12) 300deg,
            rgba(230,57,70,0.14)  330deg,
            rgba(230,57,70,0.0)   360deg
        );
    filter: blur(1px);
    opacity: 0.85;
}
.hero-rays::after {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% 55%, transparent 10%, #080808 75%);
}
.hero-content { position: relative; z-index: 2; max-width: 780px; }
.hero-eyebrow {
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.18em;
    color: rgba(255,255,255,0.5); text-transform: uppercase; margin-bottom: 1.5rem;
}
.hero-h1 {
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 800; line-height: 1.08;
    letter-spacing: -0.03em; margin: 0 0 0.5rem;
    color: #ffffff;
}
.hero-h1 em {
    font-style: italic; font-weight: 300;
    background: linear-gradient(135deg, #ff9a9e, #E63946, #c9184a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.05rem; color: rgba(255,255,255,0.55);
    font-weight: 400; line-height: 1.6; margin: 1.2rem auto 2rem;
    max-width: 540px;
}
.hero-badges {
    display: flex; gap: 0.5rem; justify-content: center;
    flex-wrap: wrap; margin-top: 2rem;
}
.hero-badge {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 50px; padding: 0.35rem 1rem;
    font-size: 0.78rem; color: rgba(255,255,255,0.7); font-weight: 500;
    backdrop-filter: blur(10px);
}

/* ── Upload Panel ─────────────────────────────────────────── */
.upload-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 2.5rem 3rem; margin: 0 3rem 2rem;
}
.upload-panel h3 {
    font-size: 1.1rem; font-weight: 700; color: white; margin-bottom: 0.3rem;
}
.upload-panel p { color: rgba(255,255,255,0.45); font-size: 0.88rem; margin-bottom: 1.5rem; }

/* ── Streamlit overrides for dark theme ──────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(255,255,255,0.15) !important;
    border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stFileUploader"] * { color: rgba(255,255,255,0.6) !important; }
.stButton > button[kind="primary"] {
    background: #E63946 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important; letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #c1121f !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(230,57,70,0.4) !important;
}
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.06) !important;
    color: white !important; border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}
.stDownloadButton > button {
    background: #E63946 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; width: 100% !important;
    padding: 0.7rem !important; font-size: 0.95rem !important;
}
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; margin: 0 3rem 1rem;
}
[data-testid="stExpander"] * { color: rgba(255,255,255,0.7) !important; }
.stStatus { background: rgba(255,255,255,0.04) !important; border-radius: 12px !important; }
.stStatus * { color: rgba(255,255,255,0.8) !important; }
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* ── Stat Cards ─────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin: 2rem 3rem; }
.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.5rem;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ""; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #E63946, #ff6b6b);
}
.stat-card .s-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.stat-card .s-val  { font-size: 2.6rem; font-weight: 900; color: #fff; line-height: 1; }
.stat-card .s-lbl  { font-size: 0.75rem; color: rgba(255,255,255,0.4); font-weight: 500; margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Section Headers ─────────────────────────────────────────── */
.sec-hdr { margin: 2.5rem 3rem 1.5rem; display: flex; align-items: center; gap: 1rem; }
.sec-hdr h2 { font-size: 1.3rem; font-weight: 800; color: #fff; margin: 0; white-space: nowrap; }
.sec-hdr hr { flex: 1; border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 0; }

/* ── Podium ──────────────────────────────────────────────────── */
.podium-wrap { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.2rem; margin: 0 3rem 1rem; }
.pod {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 1.8rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.pod:hover { border-color: rgba(230,57,70,0.4); transform: translateY(-2px); }
.pod.top1 {
    border-color: rgba(230,57,70,0.35);
    background: linear-gradient(145deg, rgba(230,57,70,0.08), rgba(255,255,255,0.03));
}
.pod .pod-medal { position: absolute; top: 1.2rem; right: 1.4rem; font-size: 1.8rem; }
.pod .pod-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 34px; height: 34px; border-radius: 50%;
    background: rgba(230,57,70,0.15); border: 1.5px solid rgba(230,57,70,0.4);
    font-size: 0.85rem; font-weight: 800; color: #E63946; margin-bottom: 1rem;
}
.pod .pod-id    { font-size: 0.7rem; color: rgba(255,255,255,0.3); letter-spacing: 0.06em; margin-bottom: 0.3rem; }
.pod .pod-title { font-size: 1rem; font-weight: 700; color: #fff; margin-bottom: 0.2rem; }
.pod .pod-co    { font-size: 0.8rem; color: rgba(255,255,255,0.4); margin-bottom: 1rem; }
.pod .pod-score { font-size: 3rem; font-weight: 900; color: #E63946; line-height: 1; }
.pod .pod-score sub { font-size: 1rem; color: rgba(255,255,255,0.25); font-weight: 400; vertical-align: super; }
.pod .bar-bg { background: rgba(255,255,255,0.08); border-radius: 50px; height: 4px; margin: 0.9rem 0 1rem; }
.pod .bar-fill { height: 4px; border-radius: 50px; background: linear-gradient(90deg,#E63946,#ff6b6b); }
.pod .chips { display: flex; flex-wrap: wrap; gap: 0.3rem; }
.pod .chip {
    background: rgba(230,57,70,0.12); color: #ff8a8a;
    border: 1px solid rgba(230,57,70,0.2); border-radius: 50px;
    padding: 0.15rem 0.65rem; font-size: 0.7rem; font-weight: 600;
}
.pod .sc-row { display: flex; gap: 1.2rem; margin-top: 0.8rem; font-size: 0.75rem; color: rgba(255,255,255,0.35); }
.pod .sc-row b { color: rgba(255,255,255,0.6); }

/* ── Table wrapper ────────────────────────────────────────────── */
.table-wrap { margin: 0 3rem 2rem; }

/* ── Download wrap ───────────────────────────────────────────── */
.dl-wrap { margin: 1rem 3rem 3rem; }

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0d0d0d !important; border-right: 1px solid rgba(255,255,255,0.07) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
section[data-testid="stSidebar"] h2 { color: white !important; font-weight: 800 !important; font-size: 1.1rem !important; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.6rem; }
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.5) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
section[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.06) !important; color: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 10px !important; font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] textarea:focus { border-color: #E63946 !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.07) !important; }
section[data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.25) !important; font-size: 0.7rem !important; }
section[data-testid="stSidebar"] [data-testid="stSlider"] * { color: white !important; }

/* Hide Streamlit default chrome */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

def is_honeypot(cand):
    expert_zero = sum(
        1 for s in cand.get("skills", [])
        if s.get("proficiency") in ["expert", "advanced"]
        and s.get("duration_months", 1) == 0
    )
    if expert_zero >= 3: return True
    for edu in cand.get("education", []):
        if edu.get("end_year", 0) < edu.get("start_year", 0): return True
    return False

def calculate_heuristic_score(cand):
    score = 1.0
    signals = cand.get("redrob_signals", {})
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr < 0.2:   score *= 0.3
    elif rr < 0.5: score *= 0.7
    last_active = signals.get("last_active_date", "2026-06-16")
    try:
        days = (datetime(2026, 6, 16) - datetime.strptime(last_active, "%Y-%m-%d")).days
        if days > 180:  score *= 0.4
        elif days > 90: score *= 0.75
    except: pass
    ic = signals.get("interview_completion_rate", 1.0)
    if ic < 0.5: score *= 0.5
    consulting = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    career = cand.get("career_history", [])
    if career and all(j.get("company", "").lower() in consulting for j in career):
        score *= 0.3
    if len(career) >= 4:
        if sum(1 for j in career if j.get("duration_months", 20) < 18) >= 4:
            score *= 0.5
    return round(score, 4)

def extract_text(cand):
    p = cand.get("profile", {})
    skills = ", ".join(s.get("name", "") for s in cand.get("skills", []))
    return f"Title: {p.get('current_title','')}. Skills: {skills}. Summary: {p.get('summary','')} "[:1000]


# ════════════════════════════════════════════════════════════════════════════
#  TOP NAV
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topnav">
    <div class="topnav-logo">
        <span class="logo-icon">r</span>
        redrob
    </div>
    <div class="topnav-links">
        <a href="#">Product</a>
        <a href="#">Solutions</a>
        <a href="#">Research</a>
        <a href="#">Pricing</a>
    </div>
    <a class="topnav-cta" href="#">Try Redrob AI ↗</a>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-rays"></div>
    <div class="hero-content">
        <div class="hero-eyebrow">⬡ India Runs · Data &amp; AI Challenge 2026</div>
        <h1 class="hero-h1">
            <em>Intelligent AI for the</em><br>
            Next Billion Professionals
        </h1>
        <p class="hero-sub">
            Upload candidate profiles and watch our semantic embedding engine
            rank, score, and surface the best-fit talent — instantly.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">🤖 all-MiniLM-L6-v2</span>
            <span class="hero-badge">⚡ FAISS Vector Search</span>
            <span class="hero-badge">🔬 Heuristic Scoring</span>
            <span class="hero-badge">🛡️ Honeypot Detection</span>
            <span class="hero-badge">📊 Real-time Ranking</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    top_n    = st.slider("Results to display", 5, 100, 20, 5)
    min_h    = st.slider("Min heuristic score", 0.0, 1.0, 0.0, 0.05)
    show_hp  = st.checkbox("Include honeypot profiles", False)
    st.markdown("---")
    st.markdown("**📋 Target Job Description**")
    jd_query = st.text_area("", height=200,
        value=(
            "AI/ML Engineer — Search & Ranking. "
            "Production experience with embedding-based retrieval: sentence-transformers, OpenAI, BGE, E5. "
            "Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS. "
            "Python expertise. Evaluation: NDCG, MRR, MAP, A/B testing. "
            "Product company preferred over consulting."
        ))
    st.markdown("---")
    st.caption("India Runs · Data & AI Challenge 2026")


# ════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️  How it works"):
    c1, c2, c3, c4 = st.columns(4)
    for col, num, title, body in [
        (c1, "01", "Parse", "Load `.json` or `.jsonl` candidate profiles from the dataset"),
        (c2, "02", "Filter", "Detect honeypots & compute multi-signal heuristic scores"),
        (c3, "03", "Embed",  "Encode text via MiniLM · FAISS cosine similarity search"),
        (c4, "04", "Rank",   "Final score = semantic similarity × heuristic score"),
    ]:
        with col:
            st.markdown(f"**{num} — {title}**\n\n{body}")


# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin:1.5rem 3rem 0.5rem;">
    <h3 style="color:white;font-weight:800;font-size:1.1rem;margin-bottom:0.3rem;">📂 Upload Candidate Profiles</h3>
    <p style="color:rgba(255,255,255,0.4);font-size:0.85rem;">Accepts <code>sample_candidates.json</code> or <code>candidates.jsonl</code></p>
</div>
""", unsafe_allow_html=True)

with st.container():
    col_up, col_btn, col_info = st.columns([3, 1, 2])
    with col_up:
        uploaded_file = st.file_uploader("", type=["jsonl", "json"], label_visibility="collapsed")
    with col_btn:
        st.markdown("<div style='padding-top:1.6rem;'>", unsafe_allow_html=True)
        go = st.button("🚀  Run Engine", type="primary", use_container_width=True,
                       disabled=uploaded_file is None)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_info:
        if uploaded_file:
            st.markdown(f"""
            <div style="padding-top:1.8rem;font-size:0.8rem;color:rgba(255,255,255,0.4);">
                📄 {uploaded_file.name} &nbsp;·&nbsp; {uploaded_file.size/1024:.1f} KB
            </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ════════════════════════════════════════════════════════════════════════════
if uploaded_file and go:
    with st.status("⚙️  Running discovery pipeline…", expanded=True) as status:
        st.write("📥  Parsing candidate file…")
        raw = uploaded_file.getvalue().decode("utf-8")
        candidates = json.loads(raw) if uploaded_file.name.endswith(".json") \
                     else [json.loads(l) for l in raw.splitlines() if l.strip()]
        total = len(candidates)

        st.write(f"🛡️  Detecting honeypots across **{total}** profiles…")
        honeypots = [c for c in candidates if is_honeypot(c)]
        pool      = candidates if show_hp else [c for c in candidates if not is_honeypot(c)]

        st.write("📊  Computing heuristic signals…")
        meta, texts = [], []
        for cand in pool:
            h = calculate_heuristic_score(cand)
            if h < min_h: continue
            texts.append(extract_text(cand))
            meta.append({
                "candidate_id": cand["candidate_id"],
                "h_score":  h,
                "title":    cand.get("profile", {}).get("current_title", "N/A"),
                "company":  cand.get("profile", {}).get("current_company", ""),
                "industry": cand.get("profile", {}).get("current_industry", ""),
                "yoe":      cand.get("profile", {}).get("years_of_experience", 0),
                "location": cand.get("profile", {}).get("location", ""),
                "skills":   [s.get("name", "") for s in cand.get("skills", [])[:5]],
            })

        st.write("🧠  Encoding embeddings + FAISS search…")
        embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([jd_query], normalize_embeddings=True)
        fidx  = faiss.IndexFlatIP(embs.shape[1])
        fidx.add(embs)
        k     = min(len(texts), top_n)
        D, I  = fidx.search(q_emb, k)
        status.update(label="✅  Pipeline complete!", state="complete")

    # Build results
    results = []
    for dist, i in zip(D[0], I[0]):
        if i == -1: continue
        m = meta[i]
        results.append({**m, "sem": float(dist), "score": round(float(dist)*m["h_score"], 4)})
    results.sort(key=lambda x: x["score"], reverse=True)

    # ── STAT CARDS ─────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-hdr"><h2>Overview</h2><hr></div>""", unsafe_allow_html=True)
    st.markdown('<div class="stat-grid">', unsafe_allow_html=True)
    for icon, val, lbl in [
        ("👥", total,          "Candidates Loaded"),
        ("🚫", len(honeypots), "Honeypots Removed"),
        ("✅", len(texts),     "Profiles Ranked"),
        ("🏆", k,              "Top Results Shown"),
    ]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="s-icon">{icon}</div>
            <div class="s-val">{val}</div>
            <div class="s-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PODIUM ─────────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-hdr"><h2>🏆 Top Candidates</h2><hr></div>""", unsafe_allow_html=True)
    st.markdown('<div class="podium-wrap">', unsafe_allow_html=True)
    medals = ["🥇", "🥈", "🥉"]
    for r, medal in zip(results[:3], medals):
        bar = int(r["score"] * 100)
        chips = "".join(f'<span class="chip">{s}</span>' for s in r["skills"][:4])
        top_cls = "top1" if medal == "🥇" else ""
        st.markdown(f"""
        <div class="pod {top_cls}">
            <div class="pod-medal">{medal}</div>
            <div class="pod-num">#{results.index(r)+1}</div>
            <div class="pod-id">{r['candidate_id']} &nbsp;·&nbsp; {r.get('location','')}</div>
            <div class="pod-title">{r['title']}</div>
            <div class="pod-co">🏢 {r.get('company','—')} &nbsp;·&nbsp; {r.get('yoe',0):.1f} yrs</div>
            <div class="pod-score">{r['score']:.3f}<sub>/ 1.0</sub></div>
            <div class="bar-bg"><div class="bar-fill" style="width:{bar}%"></div></div>
            <div class="chips">{chips}</div>
            <div class="sc-row">
                <span>Semantic <b>{r['sem']:.3f}</b></span>
                <span>Heuristic <b>{r['h_score']:.3f}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CHART ──────────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-hdr"><h2>📊 Score Distribution</h2><hr></div>""", unsafe_allow_html=True)
    chart_df = pd.DataFrame([{"Candidate": r["candidate_id"], "Score": r["score"]} for r in results]).head(20)
    chart = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#E63946").encode(
        x=alt.X("Candidate:N", sort="-y", axis=alt.Axis(labelColor="#555", labelFontSize=10, labelAngle=-40, titleColor="#555")),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelColor="#555", titleColor="#555")),
        opacity=alt.condition(alt.datum.Score > 0.5, alt.value(1.0), alt.value(0.55)),
        tooltip=["Candidate", "Score"]
    ).configure(background="transparent").configure_axis(
        gridColor="rgba(255,255,255,0.05)", domainColor="rgba(255,255,255,0.1)"
    ).properties(height=250)
    with st.container():
        st.markdown('<div style="margin:0 3rem">', unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── FULL TABLE ─────────────────────────────────────────────────────────
    st.markdown("""<div class="sec-hdr"><h2>📋 Full Ranked List</h2><hr></div>""", unsafe_allow_html=True)
    rows = [{
        "Rank": i+1, "Candidate ID": r["candidate_id"],
        "Title": r["title"], "Company": r.get("company",""),
        "Location": r.get("location",""), "YoE": round(r.get("yoe",0),1),
        "Final Score": r["score"], "Semantic": round(r["sem"],4),
        "Heuristic": r["h_score"], "Top Skills": ", ".join(r["skills"]),
    } for i,r in enumerate(results)]
    df = pd.DataFrame(rows)
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── DOWNLOAD ───────────────────────────────────────────────────────────
    sub = pd.DataFrame([{
        "candidate_id": r["candidate_id"], "rank": i+1, "score": r["score"],
        "reasoning": f"Ranked #{i+1}. Semantic {r['sem']:.3f} × heuristic {r['h_score']:.3f} = {r['score']:.4f}. {r['title']} at {r.get('company','N/A')} with {r.get('yoe',0):.1f} yrs experience."
    } for i,r in enumerate(results)])
    st.markdown('<div class="dl-wrap">', unsafe_allow_html=True)
    st.download_button("⬇️  Download submission.csv", sub.to_csv(index=False).encode("utf-8"),
                       file_name="submission.csv", mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)
