# 🏥 CVICU RAG Clinical Assistant (Prototype)

---

## 🧠 Project Summary

This project is a **local Retrieval-Augmented Generation (RAG)** assistant designed for CVICU nursing staff. It uses **vector search + local LLMs** to answer drug- and policy-related clinical questions in real-time.

Key objectives:

- Reduce time spent searching through policies
- Support onboarding of new nursing staff
- Assist clinical decision-making during critical care scenarios

---

## 📦 Architecture Overview

| Layer           | Technology                                  |
| --------------- | ------------------------------------------- |
| **Embedding**   | `sentence-transformers`                     |
| **Vector DB**   | `FAISS`                                     |
| **Chunking**    | `RecursiveCharacterTextSplitter`            |
| **LLM Backend** | `Mistral 7B` / `MedLLaMA` (GGUF, llama.cpp) |
| **Interface**   | CLI + Streamlit                             |

---

## 🔄 Pipeline Stages

1. **Extract** Markdown from nursing PDFs (via `docling`)
2. **Clean** artifacts using regex-based sanitizer
3. **Chunk** content into ~800-token segments (with 100-token overlap)
4. **Embed** chunks using `MiniLM-L6-v2`
5. **Store** embeddings in `FAISS` index
6. **Search** & **Rerank** using hybrid (cosine + keyword + header boost)
7. **Prompt** Mistral or MedLLaMA with relevant chunks for answer generation

---

## 🧪 Prompt Template

Used with `llama-cpp-python` instruct models:

```
<s>[INST] You are a helpful clinical assistant specialized in CVICU nursing policies.
Use the provided policy excerpts to answer the user's question as accurately and completely as possible.
Only use relevant clinical information such as administration, dosage, monitoring, adverse effects, precautions, and indications.
Ignore any content related to legal disclaimers, institutional boilerplate, headers, page numbers, or footnotes.

Context:
- chunk
- chunk
Question: ...
Answer: [/INST]
```

---

## 🔐 Compliance Notes

- Runs entirely locally (offline-ready)
- No patient data used
- Expandable to be HIPAA / PIPEDA compliant

---

## 🚧 Kanban Tasks

**Backlog**

- Integrate nursing policies (not just drug monographs)
- Evaluate vs commercial solutions
- Sync metadata to relational DB (e.g., PostgreSQL)

**In Progress**

- GGUF MedLLaMA integration
- Streamlit interface for real-time use
- Prompt tuning for clinical accuracy

**Review/Testing**

- Validate answers to 10+ nursing queries
- Profile speed, chunk retrieval, memory

**Done**

- End-to-end local RAG working prototype
- Heuristic reranker
- Mistral + MedLLaMA integration via `llama-cpp`
