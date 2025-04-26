# agentic_rag_chatbot/main.py

import os
import sys
import gradio as gr
from typing import Dict, Any, List, Union
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, StateGraph
from serpapi import GoogleSearch
from deepseek_api import local_llm_inference

# --- Setup vectorstore retriever ---
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "vectorstore_db",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# --- Utility ---
def get_last_message(state: Dict[str, Any]) -> str:
    if isinstance(state, tuple):
        state = state[1]
    messages = state.get("messages", [])
    if not messages:
        return ""
    return messages[-1].content.lower()

# --- Routing ---
def router_label(state: Dict[str, Any]) -> str:
    last_message = get_last_message(state)
    if any(x in last_message for x in ["search", "current", "today", "time", "weather", "news", "live"]):
        return "websearch"
    return "vectorstore"

# --- Nodes ---
def vectorstore_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, tuple):
        state = state[1]
    query = get_last_message(state)
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Answer the question based on the following documents:\n\n{context}\n\nQuestion: {query}"
    response = local_llm_inference(prompt)
    state.setdefault("messages", []).append(AIMessage(content=response))
    return state

def web_search_tool(query: str) -> str:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Error: SERPAPI_API_KEY not set."

    params = {"engine": "google", "q": query, "api_key": api_key}
    results = GoogleSearch(params).get_dict()

    if "answer_box" in results and "result" in results["answer_box"]:
        return results["answer_box"]["result"]
    if "organic_results" in results:
        return results["organic_results"][0].get("snippet", "No useful snippet found.")
    return "No relevant search results found."

def websearch_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(state, tuple):
        state = state[1]
    query = get_last_message(state)
    search_results = web_search_tool(query)
    prompt = f"Use the following search results to answer the question clearly and concisely:\n\n{search_results}\n\nQuestion: {query}"
    response = local_llm_inference(prompt)
    state.setdefault("messages", []).append(AIMessage(content=response))
    return state

# --- Build Graph ---
def build_agentic_rag_graph() -> Runnable:
    graph = StateGraph(state_schema=dict)

    def router(state):
        return (router_label(state), state)

    graph.add_node("router", router)
    graph.add_node("vectorstore", vectorstore_node)
    graph.add_node("websearch", websearch_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda out: out[0],
        {
            "vectorstore": "vectorstore",
            "websearch": "websearch"
        }
    )

    graph.add_edge("vectorstore", END)
    graph.add_edge("websearch", END)
    return graph.compile()

agent_graph = build_agentic_rag_graph()

# --- Chat CLI for testing ---
def chat_cli():
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Explain the difference between CPU and GPU in less than 3 sentences."
    messages = [HumanMessage(content=user_input)]
    state = {"messages": messages}
    result = agent_graph.invoke(state)
    if isinstance(result, tuple):
        result = result[1]
    print("\n--- LLM Response ---\n")
    print(result.get("messages", [])[-1].content if result.get("messages") else "No response.")

# --- Chat UI ---
def chat_interface(user_input: str, history: Union[List[Dict[str, str]], None]):
    messages = []
    if history:
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    if user_input.strip():
        messages.append(HumanMessage(content=user_input.strip()))

    state = {"messages": messages}
    result = agent_graph.invoke(state)

    if isinstance(result, tuple):
        result = result[1]

    new_messages = result.get("messages", [])

    gradio_history = []
    for msg in new_messages:
        if isinstance(msg, HumanMessage):
            gradio_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            gradio_history.append({"role": "assistant", "content": msg.content})

    return gradio_history

chatbot_ui = gr.ChatInterface(
    fn=chat_interface,
    title="Agentic RAG Chatbot",
    textbox=gr.Textbox(placeholder="Ask me anything...", lines=2),
    type="messages"
)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        chat_cli()
    else:
        chatbot_ui.launch(share=True)
