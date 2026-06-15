import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Redrob Candidate Discovery",
    layout="wide",
    page_icon="🎯"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-box {
    background: linear-gradient(135deg, #E63946 0%, #a01422 100%);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.hero-box h1 { font-size: 2.8rem; font-weight: 800; margin: 0; color: white; }
.hero-box p  { font-size: 1.15rem; margin-top: 0.5rem; opacity: 0.9; color: white; }

.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 5px solid #E63946;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}
.stat-card h2 { margin: 0; color: #E63946; font-size: 2rem; font-weight: 800; }
.stat-card p  { margin: 0; color: #666; font-size: 0.85rem; }

.rank-card {
    background: white;
    border-radius: 14px;
    padding: 1.5rem;
    border-top: 5px solid #E63946;
    box-shadow: 0 4px 16px rgba(230,57,70,0.12);
    margin-bottom: 1rem;
    height: 100%;
}
.rank-card .rank-badge {
    display: inline-block;
    background: #E63946;
    color: white;
    font-weight: 800;
    border-radius: 50px;
    padding: 2px 14px;
    font-size: 0.85rem;
    margin-bottom: 0.7rem;
}
.rank-card .cand-id { color: #999; font-size: 0.8rem; margin-bottom: 0.4rem; }
.rank-card .score-big { font-size: 2.2rem; font-weight: 800; color: #E63946; }
.rank-card .title-tag { font-weight: 600; color: #212529; margin-bottom: 0.5rem; }
.rank-card .reasoning { font-size: 0.88rem; color: #555; line-height: 1.5; }
.rank-card .skill-chip {
    display: inline-block;
    background: #fce8ea;
    color: #E63946;
    border-radius: 50px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin: 2px 2px 0 0;
}

.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #212529;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 3px solid #E63946;
    display: inline-block;
}

.filter-pill {
    background: #fce8ea;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


def is_honeypot(cand):
    """Detect fabricated/impossible profiles."""
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
    """Multi-signal heuristic: higher is better, never hard-filters."""
    score = 1.0
    signals = cand.get("redrob_signals", {})

    # ── Recruiter Response Rate ───────────────────────────────────────────
    rr = signals.get("recruiter_response_rate", 0.5)
    if rr < 0.2:
        score *= 0.3
    elif rr < 0.5:
        score *= 0.7

    # ── Recency ───────────────────────────────────────────────────────────
    last_active = signals.get("last_active_date", "2026-06-16")
    try:
        days_inactive = (datetime(2026, 6, 16) -
                         datetime.strptime(last_active, "%Y-%m-%d")).days
        if days_inactive > 180:
            score *= 0.4
        elif days_inactive > 90:
            score *= 0.75
    except Exception:
        pass

    # ── Interview Completion ───────────────────────────────────────────────
    ic = signals.get("interview_completion_rate", 1.0)
    if ic < 0.5:
        score *= 0.5

    # ── Consulting-only Background ────────────────────────────────────────
    consulting_firms = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
    career = cand.get("career_history", [])
    if career:
        if all(j.get("company", "").lower() in consulting_firms for j in career):
            score *= 0.3

    # ── Title-chaser (many short stints) ──────────────────────────────────
    if len(career) >= 4:
        short = sum(1 for j in career if j.get("duration_months", 20) < 18)
        if short >= 4:
            score *= 0.5

    return round(score, 4)


def extract_text(cand):
    profile = cand.get("profile", {})
    title   = profile.get("current_title", "")
    summary = profile.get("summary", "")
    skills  = ", ".join(s.get("name", "") for s in cand.get("skills", []))
    return f"Title: {title}. Skills: {skills}. Summary: {summary}"[:1000]


# ── HERO HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <h1>🎯 Redrob Intelligent Candidate Discovery</h1>
    <p>AI-Powered Semantic Ranking &amp; Heuristic Scoring Engine</p>
</div>
""", unsafe_allow_html=True)

# ── HOW IT WORKS ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    1. **Honeypot Detection** — Removes fabricated profiles (e.g., expert skills with 0 months experience).
    2. **Heuristic Scoring** — Penalises low recruiter-response rates, inactive profiles, consulting-only backgrounds, and title chasers.
    3. **Semantic Embedding** — Each candidate's title + skills + summary is embedded using `all-MiniLM-L6-v2`.
    4. **JD Matching** — Cosine similarity is computed against the *AI/ML Search & Ranking Engineer* job description.
    5. **Final Score** = `semantic_similarity × heuristic_score`
    """)

st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_n = st.slider("Number of results to show", min_value=5, max_value=100, value=20, step=5)
    show_honeypots = st.checkbox("Include honeypot candidates", value=False)
    min_h_score = st.slider("Min heuristic score threshold", 0.0, 1.0, 0.0, 0.05,
                             help="0 = show all; increase to filter weaker signals")
    st.markdown("---")
    st.markdown("**Target JD**")
    jd_query = st.text_area(
        "Job Description (editable)",
        value="AI/ML Engineer Search and Ranking. Production experience with embeddings-based retrieval like sentence-transformers, OpenAI embeddings, BGE, E5. Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS. Strong Python. Evaluation frameworks: NDCG, MRR, MAP. Product company preferred.",
        height=180
    )

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload candidates file (`.json` or `.jsonl`)",
    type=["jsonl", "json"]
)

if uploaded_file is not None:
    if st.button("🚀 Rank Candidates", type="primary"):
        with st.spinner("Processing… this may take a few seconds"):
            # Parse file
            raw = uploaded_file.getvalue().decode("utf-8")
            if uploaded_file.name.endswith(".json"):
                candidates = json.loads(raw)
            else:
                candidates = [json.loads(l) for l in raw.splitlines() if l.strip()]

            total_loaded = len(candidates)
            honeypots    = [c for c in candidates if is_honeypot(c)]
            valid        = [c for c in candidates if not is_honeypot(c)]
            if not show_honeypots:
                candidates_to_rank = valid
            else:
                candidates_to_rank = candidates

            meta, texts = [], []
            for cand in candidates_to_rank:
                h_score = calculate_heuristic_score(cand)
                if h_score < min_h_score:
                    continue
                texts.append(extract_text(cand))
                meta.append({
                    "candidate_id": cand["candidate_id"],
                    "h_score":       h_score,
                    "title":         cand.get("profile", {}).get("current_title", "N/A"),
                    "company":       cand.get("profile", {}).get("current_company", ""),
                    "skills":        [s.get("name", "") for s in cand.get("skills", [])[:4]],
                    "yoe":           cand.get("profile", {}).get("years_of_experience", 0),
                })

        if not texts:
            st.error("No candidates passed the current filters. Try lowering the heuristic threshold.")
        else:
            with st.spinner("Encoding embeddings and running FAISS search…"):
                embeddings = model.encode(texts, normalize_embeddings=True)
                q_emb      = model.encode([jd_query], normalize_embeddings=True)

                dim   = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)

                k    = min(len(texts), top_n)
                D, I = index.search(q_emb, k)

            results = []
            for dist, idx in zip(D[0], I[0]):
                if idx == -1:
                    continue
                m = meta[idx]
                final_score = float(dist) * m["h_score"]
                results.append({**m, "semantic_score": float(dist), "final_score": final_score})

            results.sort(key=lambda x: x["final_score"], reverse=True)

            # ── STATS ROW ──────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="stat-card"><h2>{total_loaded}</h2><p>Total Loaded</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="stat-card"><h2>{len(honeypots)}</h2><p>Honeypots Filtered</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-card"><h2>{len(texts)}</h2><p>Candidates Ranked</p></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="stat-card"><h2>{top_n}</h2><p>Top Shown</p></div>', unsafe_allow_html=True)

            # ── TOP 3 CARDS ────────────────────────────────────────────────
            st.markdown('<p class="section-title">🏆 Top 3 Candidates</p>', unsafe_allow_html=True)
            cols = st.columns(3)
            for col, r in zip(cols, results[:3]):
                skills_chips = "".join(f'<span class="skill-chip">{s}</span>' for s in r["skills"])
                with col:
                    st.markdown(f"""
                    <div class="rank-card">
                        <div class="rank-badge">Rank #{results.index(r)+1}</div>
                        <div class="cand-id">{r['candidate_id']} · {r.get('company','')}</div>
                        <div class="title-tag">💼 {r['title']}</div>
                        <div class="score-big">{r['final_score']:.3f}</div>
                        <div style="margin-bottom:0.6rem">{skills_chips}</div>
                        <div class="reasoning">Semantic: {r['semantic_score']:.3f} &nbsp;|&nbsp; Heuristic: {r['h_score']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── FULL TABLE ─────────────────────────────────────────────────
            st.markdown('<p class="section-title">📋 Full Ranked List</p>', unsafe_allow_html=True)
            rows = []
            for i, r in enumerate(results):
                skills_str = ", ".join(r["skills"]) or "N/A"
                rows.append({
                    "Rank":           i + 1,
                    "Candidate ID":   r["candidate_id"],
                    "Title":          r["title"],
                    "Company":        r.get("company", ""),
                    "YoE":            r.get("yoe", ""),
                    "Final Score":    round(r["final_score"], 4),
                    "Semantic Score": round(r["semantic_score"], 4),
                    "Heuristic Score":round(r["h_score"], 4),
                    "Top Skills":     skills_str,
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # ── DOWNLOAD ───────────────────────────────────────────────────
            sub_df = df[["Rank", "Candidate ID", "Final Score"]].rename(columns={
                "Rank": "rank", "Candidate ID": "candidate_id", "Final Score": "score"
            })
            sub_df["reasoning"] = [
                f"Ranked #{r+1}. Semantic similarity {s:.3f} × heuristic {h:.3f} = {f:.4f}."
                for r, s, h, f in zip(df["Rank"], df["Semantic Score"], df["Heuristic Score"], df["Final Score"])
            ]
            st.download_button(
                label="⬇️ Download submission.csv",
                data=sub_df.to_csv(index=False).encode("utf-8"),
                file_name="submission.csv",
                mime="text/csv",
            )
