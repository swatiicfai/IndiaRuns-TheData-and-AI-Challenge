import json
import gzip
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from datetime import datetime

DATA_FILE = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"
OUTPUT_INDEX = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/candidates_index.faiss"
OUTPUT_META = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/candidates_meta.json"

# We use a fast, lightweight CPU model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

def is_honeypot(cand):
    # Check for impossible skills
    expert_zero_months = sum(1 for s in cand.get("skills", []) if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 1) == 0)
    if expert_zero_months >= 3:
        return True
    
    # Check for conflicting experience
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    total_months_worked = sum(job.get("duration_months", 0) for job in cand.get("career_history", []))
    if yoe > 0 and total_months_worked == 0:
        # Might be a trap, but let's be conservative
        pass

    # A very simple check for impossible dates
    for edu in cand.get("education", []):
        if edu.get("end_year", 0) < edu.get("start_year", 0):
            return True
            
    return False

def calculate_heuristic_score(cand):
    score = 1.0
    signals = cand.get("redrob_signals", {})
    
    # Penalize low response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate < 0.2:
        score *= 0.1
    elif response_rate < 0.5:
        score *= 0.5
        
    # Penalize inactivity
    last_active = signals.get("last_active_date", "2026-06-16")
    try:
        last_active_dt = datetime.strptime(last_active, "%Y-%m-%d")
        days_inactive = (datetime(2026, 6, 16) - last_active_dt).days
        if days_inactive > 180:
            score *= 0.2
        elif days_inactive > 90:
            score *= 0.6
    except:
        pass
        
    # Penalize low interview completion
    interview_completion = signals.get("interview_completion_rate", 1.0)
    if interview_completion < 0.5:
        score *= 0.3
        
    # Check product company preference vs consulting (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    career = cand.get("career_history", [])
    all_consulting = True
    if len(career) > 0:
        for job in career:
            if job.get("company", "").lower() not in consulting_firms:
                all_consulting = False
                break
        if all_consulting:
            score *= 0.1 # Heavily penalize pure consulting
            
    # Check for title chasers (many short stints)
    if len(career) >= 4:
        short_stints = sum(1 for job in career if job.get("duration_months", 20) < 18)
        if short_stints >= 4:
            score *= 0.2 # Title chaser penalty
            
    return score

def extract_text(cand):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    skills = ", ".join([s.get("name", "") for s in cand.get("skills", [])])
    
    text = f"Title: {title}. Skills: {skills}. Summary: {summary}"
    return text[:1000] # truncate to save time/tokens

def precompute():
    print("Loading model...")
    model = SentenceTransformer(model_name)
    
    candidates_meta = []
    texts = []
    
    print("Reading candidates...")
    start_time = time.time()
    
    # We open the file
    valid_cands = 0
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            cand = json.loads(line)
            
            # Filter honeypots
            if is_honeypot(cand):
                continue
                
            h_score = calculate_heuristic_score(cand)
            if h_score < 0.05:
                # Skip totally unviable candidates to save time
                continue
                
            text = extract_text(cand)
            
            candidates_meta.append({
                "candidate_id": cand["candidate_id"],
                "h_score": h_score,
                "anonymized_name": cand.get("profile", {}).get("anonymized_name", ""),
                "current_title": cand.get("profile", {}).get("current_title", ""),
                "skills": [s.get("name") for s in cand.get("skills", [])[:5]]
            })
            texts.append(text)
            
            if len(texts) % 10000 == 0:
                print(f"Processed {len(texts)} valid candidates...")
                
    print(f"Total valid candidates: {len(texts)}")
    
    # We can encode in batches
    print("Encoding embeddings...")
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Inner product since normalized = Cosine Similarity
    index.add(embeddings)
    
    print("Saving artifacts...")
    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_META, "w", encoding='utf-8') as f:
        json.dump(candidates_meta, f)
        
    print(f"Pre-computation complete in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    precompute()
