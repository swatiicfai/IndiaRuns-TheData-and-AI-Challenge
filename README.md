# Redrob Intelligent Candidate Discovery
## Data & AI Challenge

This repository contains the solution for the "Data & AI Challenge: Intelligent Candidate Discovery" by Redrob.

### Architecture Overview
1. **Feature Extraction**: `precompute.py` parses the `candidates.jsonl` data, calculates a `heuristic_score` to heavily penalize honeypots, inactive candidates, candidates with low recruiter response rates, and consulting-heavy resumes (per the AI/ML Search & Ranking JD).
2. **Semantic Matching**: `precompute.py` embeds candidate features (Title, Skills, Summary) into a Vector index using the `sentence-transformers/all-MiniLM-L6-v2` model and stores them offline in `faiss` (`candidates_index.faiss`).
3. **Re-Ranking**: `rank.py` loads the offline index, embeds the target Job Description (Search & Ranking ML Engineer), fetches candidates from FAISS, and multiplies the cosine similarity with the `heuristic_score`. 
4. **Interactive Sandbox**: `app.py` is a Streamlit application that accepts a custom `candidates.jsonl` sample and runs the pipeline end-to-end to output the ranked candidates in CSV.

### File Structure
- `precompute.py`: Script to generate embeddings and compute heuristic scores.
- `rank.py`: Script to match against the target JD and export `submission.csv`.
- `app.py`: Streamlit Sandbox application.
- `requirements.txt`: Dependencies.

### Instructions

#### 1. Setup Environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Generate Final Ranking
```bash
# This will output submission.csv
python precompute.py
python rank.py
```

#### 3. Run Interactive Sandbox
```bash
streamlit run app.py
```
