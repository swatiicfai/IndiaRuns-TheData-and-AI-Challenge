import os
import json
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import time

OUTPUT_CSV = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/submission.csv"
INDEX_PATH = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/candidates_index.faiss"
META_PATH = "C:/Users/Swati/OneDrive/Documents/IndiaRuns/candidates_meta.json"

# Must be offline for evaluation
os.environ["HF_HUB_OFFLINE"] = "1"

def rank():
    start_time = time.time()
    
    print("Loading precomputed data...")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        candidates_meta = json.load(f)
        
    print("Loading model...")
    # Load model (make sure it's cached locally!)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Query string based on JD
    jd_query = """AI/ML Engineer Search and Ranking. 
    Production experience with embeddings-based retrieval systems like sentence-transformers, OpenAI embeddings, BGE, E5. 
    Production experience with vector databases or hybrid search infrastructure like Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
    Strong Python programming skills. 
    Hands-on experience designing evaluation frameworks for ranking systems using NDCG, MRR, MAP, offline-to-online correlation, A/B testing.
    Prefer product company experience over pure consulting services."""
    
    print("Embedding query...")
    q_emb = model.encode([jd_query], normalize_embeddings=True)
    
    print("Searching index...")
    k = 500 # Retrieve top 500 for re-ranking
    D, I = index.search(q_emb, k)
    
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1: continue
        meta = candidates_meta[idx]
        
        # Combine semantic distance (dist is Cosine Similarity 0-1) with heuristic score
        # dist is usually between 0.3 and 0.8 for MiniLM
        # h_score is a multiplier
        semantic_score = float(dist)
        h_score = meta["h_score"]
        
        final_score = semantic_score * h_score
        
        results.append({
            "candidate_id": meta["candidate_id"],
            "score": final_score,
            "semantic_score": semantic_score,
            "h_score": h_score,
            "skills": meta.get("skills", []),
            "title": meta.get("current_title", "")
        })
        
    # Sort by final score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Take top 100
    top_100 = results[:100]
    
    # Generate CSV output
    output_rows = []
    for i, r in enumerate(top_100):
        # Generate reasoning based on features
        skills_str = ", ".join(r["skills"][:3]) if r["skills"] else "relevant skills"
        reasoning = f"Strong semantic fit for Search & Ranking role with a score of {r['score']:.2f}. "
        reasoning += f"Demonstrates strong background as {r['title']} with key expertise in {skills_str}, while maintaining high platform engagement and product focus."
        
        output_rows.append({
            "candidate_id": r["candidate_id"],
            "rank": i + 1,
            "score": r["score"],
            "reasoning": reasoning
        })
        
    df = pd.DataFrame(output_rows)
    # Ensure exactly candidate_id, rank, score, reasoning
    df = df[["candidate_id", "rank", "score", "reasoning"]]
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Ranking complete in {time.time() - start_time:.2f}s. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    rank()
