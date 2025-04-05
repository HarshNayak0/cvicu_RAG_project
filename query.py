from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
import sys
import re

# === Load Embedding Model (must match training script) ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Load FAISS Index and Metadata ===
index_path = Path("rag_index.faiss")
metadata_path = Path("rag_metadata.pkl")

if not index_path.exists() or not metadata_path.exists():
    print("❌ Error: Required files 'rag_index.faiss' or 'rag_metadata.pkl' not found.")
    sys.exit(1)

index = faiss.read_index(str(index_path))
metadata = pickle.loads(metadata_path.read_bytes())


# === Normalize function to prepare vectors for cosine similarity ===
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


# === Main search function that queries FAISS and applies hybrid reranking ===
def search_chunks(query, top_k=5, boost_keywords=True, strict_file_filter=True):
    print(f"\n🔍 Searching for: {query}")

    # Embed and normalize the query
    query_vector = model.encode(query).astype("float32")
    query_vector = normalize(query_vector).reshape(1, -1)

    # Retrieve more chunks than needed to allow reranking
    distances, indices = index.search(query_vector, 15)

    # Extract keywords from the query for heuristic reranking
    query_keywords = set(re.findall(r"\w+", query.lower()))

    # Define which metadata headers are clinically relevant for scoring
    header_boost_keywords = {
        "preparation",
        "administration",
        "dosage",
        "monitoring",
        "indications",
        "contraindications",
        "precautions",
    }

    # Dynamically detect if the query targets a specific drug file
    drug_filter = None
    if strict_file_filter:
        for word in query_keywords:
            candidate = word + ".md"
            if any(m["file"].lower() == candidate for m in metadata):
                drug_filter = candidate
                break

    results = []
    for i, idx in enumerate(indices[0]):
        # Skip chunks not from the filtered drug file (if specified)
        if drug_filter and metadata[idx]["file"].lower() != drug_filter:
            continue

        result = metadata[idx].copy()
        result["distance"] = float(distances[0][i])  # vector similarity distance

        # Heuristic reranking using keyword and metadata overlap
        if boost_keywords:
            search_text = result.get("preview", "").lower()
            meta_fields = result.get("metadata", {})

            # Append header metadata and filename to search corpus
            for v in meta_fields.values():
                if isinstance(v, str):
                    search_text += " " + v.lower()
            filename = result.get("file", "").lower()
            search_text += " " + filename

            # Count matching keywords and boosted header terms
            match_score = sum(1 for word in query_keywords if word in search_text)
            header_score = sum(
                1 for word in header_boost_keywords if word in search_text
            )
            result["boost_score"] = (
                match_score + header_score * 2
            )  # Header keywords get higher weight
        else:
            result["boost_score"] = 0

        results.append(result)

    # === Sort by hybrid score combining semantic similarity and heuristic boost ===
    alpha = 2.0  # weight for keyword/header match
    beta = 1.0  # weight for vector similarity
    for r in results:
        sim_score = 1.0 - r["distance"]  # Convert distance to similarity score
        r["hybrid_score"] = alpha * r["boost_score"] + beta * sim_score

    results.sort(key=lambda x: -x["hybrid_score"])
    return results[:top_k]  # Return top k final chunks


# === Interactive Query Loop ===
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
