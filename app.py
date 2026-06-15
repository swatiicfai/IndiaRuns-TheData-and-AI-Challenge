import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime
import altair as alt

st.set_page_config(
    page_title="Redrob · Candidate Discovery",
    layout="wide",
    page_icon="🎯",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.block-container { padding-top: 0 !important; }

/* ── Hero ──────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #E63946 0%, #9B1D24 60%, #5C0F15 100%);
    border-radius: 0 0 30px 30px;
    padding: 3.5rem 2rem 3rem;
    text-align: center;
    margin: -1rem -1rem 2rem -1rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.08) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(255,255,255,0.05) 0%, transparent 50%);
}
.hero-logo {
    font-size: 0.9rem; font-weight: 700; letter-spacing: 0.25em;
    color: rgba(255,255,255,0.6); text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 3rem; font-weight: 900; color: #fff;
    line-height: 1.1; margin: 0 0 0.6rem 0;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}
.hero p {
    font-size: 1.1rem; color: rgba(255,255,255,0.75);
    margin: 0; font-weight: 400;
}
.hero-badges {
    display: flex; gap: 0.6rem; justify-content: center; margin-top: 1.4rem; flex-wrap: wrap;
}
.hero-badge {
    background: rgba(255,255,255,0.15); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.25); border-radius: 50px;
    padding: 0.3rem 1rem; font-size: 0.78rem; color: white; font-weight: 500;
}

/* ── Stat Cards ─────────────────────────────────────────────── */
.stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem; }
.stat-card {
    background: white; border-radius: 16px; padding: 1.4rem 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    border: 1px solid #f0f0f0;
    position: relative; overflow: hidden;
    transition: transform 0.2s, box-shadow 0.2s;
}
.stat-card::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, #E63946, #FF6B6B);
    border-radius: 16px 16px 0 0;
}
.stat-card .stat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.stat-card .stat-val  { font-size: 2.4rem; font-weight: 900; color: #1a1a2e; line-height: 1; }
.stat-card .stat-lbl  { font-size: 0.8rem; color: #888; font-weight: 500; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Section Headers ─────────────────────────────────────────── */
.section-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 2rem 0 1.2rem 0;
}
.section-header h3 {
    margin: 0; font-size: 1.25rem; font-weight: 800; color: #1a1a2e;
}
.section-divider {
    flex: 1; height: 1px; background: linear-gradient(90deg, #E63946 0%, transparent 100%);
}

/* ── Podium Cards ────────────────────────────────────────────── */
.podium { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.2rem; margin-bottom: 2rem; }
.pod-card {
    background: white; border-radius: 20px; padding: 1.8rem 1.5rem;
    box-shadow: 0 8px 32px rgba(230,57,70,0.1);
    border: 1px solid #f5f5f5;
    position: relative; overflow: hidden; transition: transform 0.2s;
}
.pod-card.gold   { box-shadow: 0 8px 40px rgba(230,57,70,0.18); border-color: #E63946; }
.pod-card .medal {
    position: absolute; top: 1rem; right: 1.2rem;
    font-size: 2rem; opacity: 0.85;
}
.pod-card .pod-rank {
    display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; border-radius: 50%;
    background: linear-gradient(135deg, #E63946, #FF6B6B);
    color: white; font-weight: 800; font-size: 0.9rem; margin-bottom: 1rem;
}
.pod-card .pod-id   { font-size: 0.75rem; color: #aaa; font-weight: 500; margin-bottom: 0.3rem; letter-spacing: 0.04em; }
.pod-card .pod-title{ font-size: 1rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.2rem; }
.pod-card .pod-co   { font-size: 0.82rem; color: #888; margin-bottom: 1rem; }
.pod-card .pod-score{ font-size: 2.8rem; font-weight: 900; color: #E63946; line-height: 1; }
.pod-card .pod-score span { font-size: 0.95rem; color: #bbb; font-weight: 400; }
.pod-card .score-bar-bg {
    background: #f5f5f5; border-radius: 50px; height: 6px; margin: 0.8rem 0;
}
.pod-card .score-bar-fill {
    height: 6px; border-radius: 50px;
    background: linear-gradient(90deg, #E63946, #FF6B6B);
}
.pod-card .chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.8rem; }
.pod-card .chip {
    background: #fff0f1; color: #E63946; border-radius: 50px;
    padding: 0.2rem 0.7rem; font-size: 0.72rem; font-weight: 600;
    border: 1px solid #ffd0d3;
}
.pod-card .score-row {
    display: flex; gap: 1rem; margin-top: 0.8rem; font-size: 0.78rem; color: #888;
}
.pod-card .score-row span b { color: #555; }

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #1a1a2e !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSlider > div > div { background: #E63946 !important; }
section[data-testid="stSidebar"] textarea {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] textarea:focus {
    border-color: #E63946 !important;
    box-shadow: 0 0 0 2px rgba(230,57,70,0.3) !important;
}
section[data-testid="stSidebar"] label { color: rgba(255,255,255,0.7) !important; font-size: 0.8rem !important; }
section[data-testid="stSidebar"] .stCheckbox label { color: white !important; font-size: 0.9rem !important; }
section[data-testid="stSidebar"] h2 { font-size: 1.1rem !important; font-weight: 700 !important; border-bottom: 1px solid rgba(255,255,255,0.15); padding-bottom: 0.5rem; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }
section[data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.4) !important; font-size: 0.72rem !important; }

/* ── Upload Area ──────────────────────────────────────────────── */
.upload-wrapper {
    background: white; border-radius: 16px; padding: 1.5rem 2rem;
    border: 2px dashed #f0d0d2; box-shadow: 0 2px 16px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}
.upload-hint { font-size: 0.85rem; color: #aaa; margin-top: 0.5rem; }

/* ── Download Button ──────────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #E63946, #9B1D24) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    padding: 0.7rem 1.8rem !important; font-size: 1rem !important;
    width: 100% !important;
}

/* ── Rank Button ──────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #E63946, #9B1D24) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    padding: 0.7rem 2rem !important; font-size: 1rem !important;
    letter-spacing: 0.03em !important;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(230,57,70,0.35) !important; }

/* ── Expander ─────────────────────────────────────────────────── */
.streamlit-expanderHeader { font-weight: 600 !important; }
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
    if expert_zero >= 3:
        return True
    for edu in cand.get("education", []):
        if edu.get("end_year", 0) < edu.get("start_year", 0):
            return True
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
    except Exception: pass
    ic = signals.get("interview_completion_rate", 1.0)
    if ic < 0.5: score *= 0.5
    consulting = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    career = cand.get("career_history", [])
    if career and all(j.get("company", "").lower() in consulting for j in career):
        score *= 0.3
    if len(career) >= 4:
        short = sum(1 for j in career if j.get("duration_months", 20) < 18)
        if short >= 4: score *= 0.5
    return round(score, 4)

def extract_text(cand):
    p = cand.get("profile", {})
    title   = p.get("current_title", "")
    summary = p.get("summary", "")
    skills  = ", ".join(s.get("name", "") for s in cand.get("skills", []))
    return f"Title: {title}. Skills: {skills}. Summary: {summary}"[:1000]


# ════════════════════════════════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-logo">⬡ Redrob Platform</div>
    <h1>Intelligent Candidate Discovery</h1>
    <p>Semantic Embedding Search · Heuristic Scoring · Honeypot Filtering</p>
    <div class="hero-badges">
        <span class="hero-badge">🤖 all-MiniLM-L6-v2</span>
        <span class="hero-badge">⚡ FAISS Vector Search</span>
        <span class="hero-badge">🔬 Multi-Signal Ranking</span>
        <span class="hero-badge">🛡️ Honeypot Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    top_n = st.slider("Results to display", 5, 100, 20, 5)
    min_h = st.slider("Min heuristic score", 0.0, 1.0, 0.0, 0.05)
    show_hp = st.checkbox("Show honeypot profiles", False)
    st.markdown("---")
    st.markdown("**📋 Target Job Description**")
    jd_query = st.text_area(
        "", height=220,
        value=(
            "AI/ML Engineer — Search & Ranking. "
            "Production experience with embedding-based retrieval: sentence-transformers, OpenAI, BGE, E5. "
            "Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch. "
            "Python expertise. Evaluation: NDCG, MRR, MAP, A/B testing. "
            "Product company preferred over consulting."
        )
    )
    st.markdown("---")
    st.caption("India Runs · Data & AI Challenge 2026")


# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ How it works", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a: st.info("**① Parse**\nLoad `.json` or `.jsonl` candidate profiles")
    with col_b: st.info("**② Filter**\nDetect honeypots & score heuristics")
    with col_c: st.info("**③ Embed**\nEncode text via MiniLM · FAISS search")
    with col_d: st.info("**④ Rank**\nFinal score = semantic × heuristic")

uploaded_file = st.file_uploader(
    "📂  Upload `sample_candidates.json` or `candidates.jsonl`",
    type=["jsonl", "json"]
)

if uploaded_file:
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        go = st.button("🚀  Run Discovery Engine", type="primary", use_container_width=True)
    with col_note:
        st.caption(f"File: **{uploaded_file.name}** · {uploaded_file.size/1024:.1f} KB")

    if go:
        # ── Parse ──────────────────────────────────────────────────────────
        with st.status("⚙️ Running pipeline…", expanded=True) as status_box:
            st.write("📥 Parsing candidate file…")
            raw = uploaded_file.getvalue().decode("utf-8")
            candidates = json.loads(raw) if uploaded_file.name.endswith(".json") \
                         else [json.loads(l) for l in raw.splitlines() if l.strip()]
            total = len(candidates)

            st.write(f"🛡️ Detecting honeypots across {total} profiles…")
            honeypots = [c for c in candidates if is_honeypot(c)]
            valid     = [c for c in candidates if not is_honeypot(c)]
            pool      = candidates if show_hp else valid

            st.write("📊 Computing heuristic signals…")
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
                    "response_rate": cand.get("redrob_signals", {}).get("recruiter_response_rate", None),
                })

            st.write("🧠 Encoding embeddings + FAISS nearest-neighbour search…")
            embs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            q_emb = model.encode([jd_query], normalize_embeddings=True)
            idx   = faiss.IndexFlatIP(embs.shape[1])
            idx.add(embs)
            k     = min(len(texts), top_n)
            D, I  = idx.search(q_emb, k)
            status_box.update(label="✅ Pipeline complete!", state="complete")

        # ── Build results ──────────────────────────────────────────────────
        results = []
        for dist, i in zip(D[0], I[0]):
            if i == -1: continue
            m = meta[i]
            results.append({**m, "sem": float(dist), "score": float(dist) * m["h_score"]})
        results.sort(key=lambda x: x["score"], reverse=True)

        # ── STAT CARDS ──────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "👥", total,          "Candidates Loaded"),
            (c2, "🚫", len(honeypots), "Honeypots Removed"),
            (c3, "✅", len(texts),     "Profiles Ranked"),
            (c4, "🏆", k,             "Top Results Shown"),
        ]
        for col, icon, val, lbl in cards:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon">{icon}</div>
                    <div class="stat-val">{val}</div>
                    <div class="stat-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        # ── PODIUM ──────────────────────────────────────────────────────────
        medals = ["🥇", "🥈", "🥉"]
        st.markdown("""
        <div class="section-header">
            <h3>🏆 Top Candidates</h3>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)

        cols3 = st.columns(3)
        for col, r, medal in zip(cols3, results[:3], medals):
            bar_w = int(r["score"] * 100)
            chips = "".join(f'<span class="chip">{s}</span>' for s in r["skills"][:4])
            gold_cls = "gold" if medal == "🥇" else ""
            with col:
                st.markdown(f"""
                <div class="pod-card {gold_cls}">
                    <div class="medal">{medal}</div>
                    <div class="pod-rank">#{results.index(r)+1}</div>
                    <div class="pod-id">{r['candidate_id']} · {r.get('location','')}</div>
                    <div class="pod-title">{r['title']}</div>
                    <div class="pod-co">🏢 {r.get('company','—')} &nbsp;·&nbsp; 📅 {r.get('yoe',0):.1f} yrs</div>
                    <div class="pod-score">{r['score']:.3f}<span> / 1.0</span></div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_w}%"></div>
                    </div>
                    <div class="chips">{chips}</div>
                    <div class="score-row">
                        <span>Semantic <b>{r['sem']:.3f}</b></span>
                        <span>Heuristic <b>{r['h_score']:.3f}</b></span>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── SCORE DISTRIBUTION CHART ─────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            <h3>📊 Score Distribution</h3>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)

        chart_df = pd.DataFrame([
            {"Candidate": r["candidate_id"], "Score": round(r["score"], 4),
             "Semantic": round(r["sem"], 4), "Heuristic": round(r["h_score"], 4)}
            for r in results
        ])
        bar_chart = alt.Chart(chart_df.head(20)).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("Candidate:N", sort="-y", axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Score:Q", scale=alt.Scale(scheme="reds"), legend=None),
            tooltip=["Candidate", "Score", "Semantic", "Heuristic"]
        ).properties(height=280)
        st.altair_chart(bar_chart, use_container_width=True)

        # ── FULL TABLE ──────────────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
            <h3>📋 Full Ranked List</h3>
            <div class="section-divider"></div>
        </div>""", unsafe_allow_html=True)

        rows = [{
            "Rank":           i + 1,
            "Candidate ID":   r["candidate_id"],
            "Title":          r["title"],
            "Company":        r.get("company", ""),
            "Location":       r.get("location", ""),
            "YoE":            r.get("yoe", ""),
            "Final Score":    round(r["score"], 4),
            "Semantic":       round(r["sem"], 4),
            "Heuristic":      round(r["h_score"], 4),
            "Top Skills":     ", ".join(r["skills"]),
        } for i, r in enumerate(results)]

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.background_gradient(subset=["Final Score"], cmap="Reds"),
            use_container_width=True, hide_index=True, height=420
        )

        # ── DOWNLOAD ────────────────────────────────────────────────────────
        st.markdown("---")
        sub = pd.DataFrame([{
            "candidate_id": r["candidate_id"],
            "rank":         i + 1,
            "score":        round(r["score"], 4),
            "reasoning":    (
                f"Ranked #{i+1}. Semantic match {r['sem']:.3f} × heuristic {r['h_score']:.3f} = {r['score']:.4f}. "
                f"Profile: {r['title']} at {r.get('company','N/A')} with {r.get('yoe',0):.1f} years experience."
            )
        } for i, r in enumerate(results)])

        st.download_button(
            "⬇️  Download submission.csv",
            sub.to_csv(index=False).encode("utf-8"),
            file_name="submission.csv",
            mime="text/csv",
            use_container_width=True
        )
