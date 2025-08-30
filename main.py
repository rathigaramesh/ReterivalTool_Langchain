import gradio as gr
from retrievers.utils import load_and_split
from retrievers.vector_retriever import build_vector_retriever
from retrievers.bm25_retriever import build_bm25_retriever
from retrievers.hybrid_retriever import build_hybrid_retriever
from retrievers.multiquery_retriever import build_multiquery_retriever
from pipeline_chain import run_chain_mode
from pipeline_agent import build_agent, run_agent_mode
from config import GOOGLE_API_KEY

# Load docs
chunks = load_and_split("S:/AI_RELATED/AI_PROJECTS/Reteriver_LangchainTool/data/datatest.txt")

# Build retrievers
vector_retriever = build_vector_retriever(chunks, GOOGLE_API_KEY)
bm25_retriever = build_bm25_retriever(chunks)
hybrid_retriever = build_hybrid_retriever(vector_retriever, bm25_retriever)
multiquery_retriever = build_multiquery_retriever(vector_retriever, GOOGLE_API_KEY)

retrievers_map = {
    "Vector": vector_retriever,
    "BM25": bm25_retriever,
    "Hybrid": hybrid_retriever,
    "MultiQuery": multiquery_retriever
}

agent = build_agent(retrievers_map)

def ask_question(query, retriever_choice, mode_choice):
    if mode_choice == "Chain Mode":
        answer, sources = run_chain_mode(query, retrievers_map[retriever_choice])
        return f"Answer:\n{answer}\n\nSources:\n" + "\n---\n".join(sources)
    else:
        return run_agent_mode(query, agent)

iface = gr.Interface(
    fn=ask_question,
    inputs=[
        "text",
        gr.Dropdown(["Vector", "BM25", "Hybrid", "MultiQuery"], label="Retriever (Chain Mode only)"),
        gr.Radio(["Chain Mode", "Agent Mode"], value="Chain Mode", label="Execution Mode")
    ],
    outputs="text",
    title="Retriever Playground",
    description="Try LangChain retrievers in Chain mode or Agent mode (Google Gemini)."
)

if __name__ == "__main__":
    iface.launch()
