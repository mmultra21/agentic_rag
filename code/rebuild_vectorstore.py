# rebuild_vectorstore.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader("/project/data/sample_docs.txt")  # <-- adjust your source file
docs = loader.load()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed with HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save locally
vectorstore.save_local("vectorstore_db")
print("✅ FAISS vectorstore created and saved to 'vectorstore_db/'")
