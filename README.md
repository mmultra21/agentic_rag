[![GitHub Repo](https://img.shields.io/badge/GitHub-agentic_rag-blue?logo=github)](https://github.com/mmultra21/agentic_rag)

# Agentic RAG Chatbot

[![GitHub Repo](https://img.shields.io/badge/GitHub-agentic_rag-blue?logo=github)](https://github.com/mmultra21/agentic_rag)

## 🌍 Overview
The **Agentic RAG Chatbot** is a powerful, modular retrieval-augmented generation (RAG) system designed to perform intelligent document retrieval and interactive conversation using large language models (LLMs). It leverages agentic behavior to autonomously guide conversations, enrich responses, and interact with external knowledge bases when necessary.

---

## 🔍 Key Features
- **Retrieval-Augmented Generation (RAG)**: Combines real-time retrieval of external documents with LLM generation.
- **Agentic Behavior**: Embeds decision-making strategies to plan, retrieve, and respond intelligently.
- **Customizable Models**: Supports integration with local and cloud-based LLMs.
- **Scalable Design**: Suitable for enterprise-scale document bases.
- **GPU Ready**: Optimized for environments like NVIDIA AI Workbench.

---

## 🛠️ Project Structure
```bash
agentic_rag/
├── models/                  # (Large models excluded from GitHub)
├── code/                     # Core logic, retrieval, chatbot agents
├── configs/                  # Configuration files for models, embeddings, retrieval settings
├── data/                     # Sample datasets or indexing data
├── README.md                 # This file
└── requirements.txt          # Python dependencies
```

---

## 🔄 Installation

```bash
# Clone the repo
git clone https://github.com/mmultra21/agentic_rag.git
cd agentic_rag

# Install required packages
pip install -r requirements.txt
```

---

## 🤖 Usage

1. **Configure** your model and retrieval settings in `configs/`.
2. **Launch** the chatbot:

```bash
python code/main_chatbot.py
```

3. **Interact** naturally with the agent, asking questions based on your indexed knowledge base.

---

## 🎉 Coming Soon
- Fine-tuned agent policies for domain-specific tasks
- Integration with advanced vector databases (e.g., FAISS, Milvus)
- Web frontend for live chatting

---

## 👥 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## ❤️ Credits
This project was developed as part of NVIDIA AI Workbench workflows.
Special thanks to the open-source LLM and RAG communities!

---

## 🚀 License
[MIT License](LICENSE)

---

> **Note:** Large model files (e.g., `deepseek-llm-7b-chat.Q4_K_M.gguf`) are excluded from GitHub for compliance with repository size limits. Please download required models manually.



