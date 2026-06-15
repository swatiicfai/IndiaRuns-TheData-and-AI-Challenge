import streamlit as st
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Candidate Discovery Sandbox", layout="wide")

@st.cache_resource
def load_model():
    # Cache the model to avoid reloading
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

def is_honeypot(cand):
    expert_zero_months = sum(1 for s in cand.get("skills", []) if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 1) == 0)
    if expert_zero_months >= 3: return True
    for edu in cand.get("education", []):
        if edu.get("end_year", 0) < edu.get("start_year", 0): return True
    return False

def calculate_heuristic_score(cand):
    score = 1.0
    signals = cand.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.2: score *= 0.1
    elif response_rate < 0.5: score *= 0.5
    last_active = signals.get("last_active_date", "2026-06-16")
    try:
        last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
        days_inactive = (datetime(2026, 6, 16) - last_active_dt).days
        if days_inactive > 180: score *= 0.2
        elif days_inactive > 90: score *= 0.6
    except: pass
    interview_completion = signals.get("interview_completion_rate", 1.0)
    if interview_completion < 0.5: score *= 0.3
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    career = cand.get("career_history", [])
    if len(career) > 0:
        all_consulting = all(job.get("company", "").lower() in consulting_firms for job in career)
        if all_consulting: score *= 0.1
    if len(career) >= 4:
        short_stints = sum(1 for job in career if job.get("duration_months", 20) < 18)
        if short_stints >= 4: score *= 0.2
    return score

def extract_text(cand):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    skills = ", ".join([s.get("name", "") for s in cand.get("skills", [])])
    return f"Title: {title}. Skills: {skills}. Summary: {summary}"[:1000]

st.markdown("""
<div style="text-align: center; padding-bottom: 2rem;">
    <h1 style="color: #FF4B4B; font-size: 3rem;">Redrob Intelligent Discovery 🚀</h1>
    <p style="font-size: 1.2rem; color: #666;">AI-Powered Candidate Ranking & Semantic Matching Engine</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
### 🎯 How it works
This sandbox mimics our offline batch processor. It filters honeypot profiles, computes heuristic scores based on metadata, and executes a semantic embedding search against the **AI/ML Search & Ranking Engineer** job description.
""")

uploaded_file = st.file_uploader("Upload candidates sample", type=["jsonl", "json"])

if uploaded_file is not None:
    if st.button("Rank Candidates"):
        with st.spinner("Processing candidates..."):
            file_content = uploaded_file.getvalue().decode('utf-8')
            if uploaded_file.name.endswith('.json'):
                candidates = json.loads(file_content)
            else:
                candidates = [json.loads(line) for line in file_content.splitlines() if line.strip()]
            
            st.write(f"Loaded {len(candidates)} candidates.")
            
            meta = []
            texts = []
            for cand in candidates:
                if is_honeypot(cand): continue
                h_score = calculate_heuristic_score(cand)
                if h_score < 0.05: continue
                texts.append(extract_text(cand))
                meta.append({
                    "candidate_id": cand["candidate_id"],
                    "h_score": h_score,
                    "title": cand.get("profile", {}).get("current_title", ""),
                    "skills": [s.get("name") for s in cand.get("skills", [])[:3]]
                })
                
            if len(texts) == 0:
                st.error("No valid candidates found after filtering.")
            else:
                embeddings = model.encode(texts, normalize_embeddings=True)
                jd_query = "AI/ML Engineer Search and Ranking. Production experience with embeddings-based retrieval systems like sentence-transformers, OpenAI embeddings, BGE, E5. Production experience with vector databases like Pinecone, Weaviate, Qdrant, Milvus. Strong Python programming skills. Hands-on experience evaluating ranking systems using NDCG, MRR, MAP. Prefer product company experience over pure consulting services."
                q_emb = model.encode([jd_query], normalize_embeddings=True)
                
                # Manual dot product for small scale instead of faiss to save dependencies in pure streamlit if needed
                # But we have faiss, so let's use it
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                
                k = min(100, len(texts))
                D, I = index.search(q_emb, k)
                
                results = []
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1: continue
                    m = meta[idx]
                    semantic_score = float(dist)
                    final_score = semantic_score * m["h_score"]
                    results.append({
                        "candidate_id": m["candidate_id"],
                        "score": final_score,
                        "title": m["title"],
                        "skills": m["skills"]
                    })
                    
                results.sort(key=lambda x: x["score"], reverse=True)
                
                output_rows = []
                for i, r in enumerate(results):
                    skills_str = ", ".join(r["skills"]) if r["skills"] else "relevant skills"
                    reasoning = f"Strong semantic fit for Search & Ranking role with a score of {r['score']:.2f}. Demonstrates strong background as {r['title']} with key expertise in {skills_str}."
                    output_rows.append({
                        "candidate_id": r["candidate_id"],
                        "rank": i + 1,
                        "score": r["score"],
                        "reasoning": reasoning
                    })
                    
                df = pd.DataFrame(output_rows)
                st.success("Ranking complete! 🎉")
                st.markdown("### 🏆 Top 3 Candidates")
                
                cols = st.columns(3)
                for c, r in zip(cols, output_rows[:3]):
                    with c:
                        st.markdown(f'''
                        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-top: 5px solid #FF4B4B; margin-bottom: 20px;">
                            <h4 style="margin-top: 0;">Rank #{r['rank']}</h4>
                            <p style="color:gray; font-size: 0.8rem; margin-bottom: 5px;">ID: {r['candidate_id']}</p>
                            <h2 style="color: #FF4B4B; margin: 10px 0;">{r['score']:.2f} <span style="font-size:1rem;color:gray;font-weight:normal;">/ 1.0</span></h2>
                            <p style="font-size: 0.9rem;">{r['reasoning']}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown("### 📋 Full Ranked List")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download submission.csv",
                    data=csv,
                    file_name='submission.csv',
                    mime='text/csv',
                )
