import streamlit as st
import sys
import os
from llama_cpp import Llama
from query import search_chunks

# Model configuration
MODELS = {
    "Mistral-7B Instruct": {
        "path": "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
        "n_gpu_layers": 0,
        "description": "General-purpose instruction-tuned model",
    },
    "MedLLaMA": {
        "path": "models/MedLLaMA-3.Q4_K_M.gguf",
        "n_gpu_layers": 20,
        "description": "Medical-tuned model for clinical relevance",
    },
}

# Lazy-load LLM models
@st.cache_resource
def load_model(model_name):
    config = MODELS[model_name]
    return Llama(
        model_path=config["path"],
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=config["n_gpu_layers"],
        verbose=False,
    )

# Build formatted prompt from chunks and question
def build_prompt(query, chunks):
    context = "\n\n".join(
        f"File: {chunk['file']} | Chunk ID: {chunk['chunk_id']}\n{chunk['preview']}"
        for chunk in chunks
    )
    system_prompt = (
        "You are a helpful clinical assistant specialized in CVICU nursing policies. "
        "Use the provided policy excerpts to answer the user's question as accurately and completely as possible. "
        "Only use relevant clinical information such as administration, dosage, monitoring, adverse effects, precautions, and indications. "
        "Include exact dosages, infusion details, and monitoring instructions when provided. "
        "Ignore any content related to legal disclaimers, institutional boilerplate, headers, page numbers, or footnotes. "
        "If the answer is not directly stated, use your best clinical judgment based on the context."
    )
    return f"<s>[INST] {system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer: [/INST]"

# Run query and generate answer
def generate_answer(llm, query):
    top_chunks = search_chunks(query)
    prompt = build_prompt(query, top_chunks)
    output = llm(prompt, max_tokens=350, temperature=0.7)
    answer = output.get("choices", [{}])[0].get("text", "").strip()
    if not answer:
        answer = "[⚠️ Model returned no response.]"
    return answer, top_chunks

# Streamlit app UI
def main():
    st.set_page_config(page_title="CVICU Assistant", layout="wide")
    st.title("🩺 CVICU Clinical Assistant")
    st.markdown("Query Sunnybrook CVICU nursing policy documents using local LLMs.")

    with st.sidebar:
        st.header("⚙️ Settings")
        selected_model = st.selectbox("Select Model", list(MODELS.keys()))
        st.caption(MODELS[selected_model]["description"])
        if st.button("❌ Exit App"):
            st.stop()
            st.write("Shutting Down...")
            sys.exit()

    llm = load_model(selected_model)

    query = st.text_input("💬 Enter a clinical question:")
    if st.button("🧠 Generate Answer") and query:
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(llm, query)
        st.subheader("🩺 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for chunk in sources:
            st.markdown(f"- **{chunk['file']}** (Chunk ID: `{chunk['chunk_id']}`)")

if __name__ == "__main__":
    main()
