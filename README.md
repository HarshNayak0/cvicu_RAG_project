
# 🏥 CVICU RAG Clinical Assistant (Prototype)

---

## 🧠 Project Summary

This project is a **local retrieval-augmented generation (RAG)** assistant for CVICU nursing drug monographs. It uses **vector search + local LLM** to answer clinical questions in real-time. Designed to reduce policy lookup time, onboard new staff, and support clinical decision-making.

---

## 📦 Architecture Overview

| Layer         | Technology                   |
|---------------|-------------------------------|
| Embedding     | sentence-transformers         |
| Vector DB     | FAISS                         |
| Chunking      | RecursiveCharacterTextSplitter |
| LLM Backend   | Mistral 7B / MedLLaMA         |
| Interface     | CLI (for now)                 |

---

## 🔄 Pipeline Stages

1. **Extract & Clean** PDFs (Docling)
2. **Chunk** into 800-token blocks
3. **Embed** via MiniLM
4. **Store** in FAISS
5. **Search + Rerank** chunks for query
6. **Send prompt** to Mistral/MedLLaMA
7. **Return clinically filtered answer**

---

## 🧪 Prompt Template

> Used for context-based generation:

```
You are a helpful clinical assistant...
Context:
- chunk
- chunk
Question: ...
Answer:
```

---

## 🔐 Compliance Notes

- Designed for local deployment
- No patient data used
- Expandable to be HIPAA / PIPEDA compliant

---

## 🚧 Active Tasks (Kanban)

**View as Board**

### Backlog
- Integrate nursing policies (not just drug monographs)
- Add frontend (Streamlit or Flask)
- Evaluate vs commercial tools
- Sync metadata to relational DB (PostgreSQL)

### In Progress
- Migrate to MedLLaMA with GGUF
- Optimize prompt building for precision
- Evaluate on more drug examples

### Review/Testing
- Clinical accuracy for 10 drug questions
- Performance profiling (token eval time, chunk retrieval quality)

### Done
- Basic extraction/cleaning pipeline
- Cosine similarity w/ keyword reranking
- Mistral 7B local integration via llama-cpp
