import os
import gradio as gr
from typing import Dict, Any
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langchain_community.tools import Tool
from serpapi import GoogleSearch
import requests

# Direct function replacement from deepseek.py

def local_llm_inference(prompt: str, temperature: float = 0.7) -> str:
    try:
        response = requests.post(
            "http://localhost:11435/generate",
            json={"prompt": prompt, "temperature": temperature},
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("response", "[No response]").strip()
    except Exception as e:
        return f"[LLM Error] {e}"

# Load the vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("vectorstore_db", embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Define nodes
def router(state: Dict[str, Any]) -> str:
    last_message = state["messages"][-1].content.lower()
    if any(x in last_message for x in ["search", "current", "today", "time", "weather", "news", "live"]):
        return "websearch"
    return "vectorstore"

def vectorstore_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following documents:\n\n{context}\n\nQuestion: {query}"
    response = local_llm_inference(prompt)
    state["messages"].append(AIMessage(content=response))
    return state

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["messages"][-1].content
    search_results = web_search_tool(query)
    prompt = f"Use the following search results to answer the question clearly and concisely:\n\n{search_results}\n\nQuestion: {query}"
    response = local_llm_inference(prompt)
    state["messages"].append(AIMessage(content=response))
    return state

def web_search_tool(query: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set."

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "answer_box" in results and "result" in results["answer_box"]:
        return results["answer_box"]["result"]

    if "organic_results" in results:
        return results["organic_results"][0].get("snippet", "No useful snippet found.")

    return "No relevant search results found."

# Build LangGraph
class AgentState(Dict[str, Any]):
    messages: list

def build_agentic_rag_graph() -> Runnable:
    builder = StateGraph()
    builder.add_node("router", router)
    builder.add_node("vectorstore", vectorstore_node)
    builder.add_node("websearch", websearch_node)
    builder.set_entry_point("router")
    builder.add_conditional_edges("router", router, {
        "vectorstore": "vectorstore",
        "websearch": "websearch"
    })
    builder.add_edge("vectorstore", END)
    builder.add_edge("websearch", END)
    return builder.compile()

agent_graph = build_agentic_rag_graph()

def chat_interface(user_input, history):
    messages = history if isinstance(history, list) else []
    state = {"messages": messages + [HumanMessage(content=user_input)]}
    new_state = agent_graph.invoke(state)
    response = new_state["messages"][-1].content
    return new_state["messages"], new_state["messages"]

chatbot_ui = gr.ChatInterface(
    fn=chat_interface,
    title="Agentic RAG Chatbot",
    textbox=gr.Textbox(placeholder="Ask me anything...", lines=2),
    type="chatbot"
)

if __name__ == "__main__":
    chatbot_ui.launch(share=True)

