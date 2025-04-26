from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os
import subprocess

# Define request schema
class RequestData(BaseModel):
    prompt: str
    temperature: float = 0.7

# Initialize FastAPI app
app = FastAPI(title="DeepSeek Local API")

# Absolute path to your model file
MODEL_PATH = "/project/models/deepseek-llm-7b-chat.Q4_K_M.gguf"

# Validate model presence
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Auto-detect CUDA availability
def is_cuda_available():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except FileNotFoundError:
        return False

USE_CUDA = is_cuda_available()
print(f"CUDA Detected: {USE_CUDA}")

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,               # Use full context length
    n_threads=os.cpu_count(),  # Use all available CPU threads
    n_gpu_layers=-1 if USE_CUDA else 0,  # Load fully onto GPU if CUDA is detected
    use_mlock=True,
    use_mmap=True,
    verbose=True
)

@app.post("/generate")
def generate_text(request: RequestData):
    try:
        prompt = f"You are a helpful assistant. Only respond in English.\n\n{request.prompt}"
        result = llm(
            prompt=prompt,
            max_tokens=768,
            temperature=request.temperature,
            top_p=0.95,
            stop=["<|eot_id|>"]
        )
        return {"response": result["choices"][0]["text"].strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Expose function to be used from other files (like agentic_rag.py)
def local_llm_inference(prompt: str) -> str:
    import requests
    try:
        res = requests.post("http://localhost:11437/generate", json={"prompt": prompt})
        return res.json().get("response", "No response.")
    except Exception as e:
        return f"Error: {e}"

# Run server if script is directly executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11437)
