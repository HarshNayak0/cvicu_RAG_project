from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import sys
import re

# Load embedding model (must match model used in test.py)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata safely
index_path = Path("rag_index.faiss")
metadata_path = Path("rag_metadata.pkl")

if not index_path.exists() or not metadata_path.exists():
    print("❌ Error: Required files 'rag_index.faiss' or 'rag_metadata.pkl' not found.")
    sys.exit(1)

index = faiss.read_index(str(index_path))
metadata = pickle.loads(metadata_path.read_bytes())

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def search_chunks(query, top_k=5, boost_keywords=True, strict_file_filter=True):
    print(f"\n🔍 Searching for: {query}")

    # Embed and normalize the query for cosine similarity
    query_vector = model.encode(query).astype("float32")
    query_vector = normalize(query_vector).reshape(1, -1)

    # Search the FAISS index
    distances, indices = index.search(query_vector, 15)  # Retrieve more to rerank

    # Clean and split query for basic keyword matching
    query_keywords = set(re.findall(r"\w+", query.lower()))

    # Headers worth boosting in scoring
    header_boost_keywords = {"preparation", "administration", "dosage", "monitoring", "indications", "contraindications", "precautions"}

    # Determine if a drug-specific filter should be used
    drug_filter = None
    if strict_file_filter:
        for word in query_keywords:
            candidate = word + ".md"
            if any(m["file"].lower() == candidate for m in metadata):
                drug_filter = candidate
                break

    results = []
    for i, idx in enumerate(indices[0]):
        if drug_filter and metadata[idx]["file"].lower() != drug_filter:
            continue
        result = metadata[idx].copy()
        result["distance"] = float(distances[0][i])

        # Dynamic boosting using keyword overlap
        if boost_keywords:
            search_text = result.get("preview", "").lower()
            meta_fields = result.get("metadata", {})
            for v in meta_fields.values():
                if isinstance(v, str):
                    search_text += " " + v.lower()
            filename = result.get("file", "").lower()
            search_text += " " + filename

            match_score = sum(1 for word in query_keywords if word in search_text)
            header_score = sum(1 for word in header_boost_keywords if word in search_text)
            result["boost_score"] = match_score + header_score * 2  # Heavily weight headers
        else:
            result["boost_score"] = 0

        results.append(result)

    # Sort by hybrid score
    alpha = 2.0  # keyword boost
    beta = 1.0   # vector sim (1 - distance)
    for r in results:
        sim_score = 1.0 - r["distance"]
        r["hybrid_score"] = alpha * r["boost_score"] + beta * sim_score

    results.sort(key=lambda x: -x["hybrid_score"])
    return results[:top_k]

if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        top_chunks = search_chunks(query)

        print("\nTop Results:\n")
        for i, chunk in enumerate(top_chunks):
            print(f"--- Result {i+1} ---")
            print(f"File: {chunk['file']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Distance: {chunk['distance']:.4f}")
            print(f"Keyword Match Score: {chunk['boost_score']}")
            print(f"Preview: {chunk['preview']}\n")
