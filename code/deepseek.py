from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import os

# Define request schema
class RequestData(BaseModel):
    prompt: str
    temperature: float = 0.7

# Create FastAPI app
app = FastAPI(title="DeepSeek Local API")

# Absolute path to your model
MODEL_PATH = "/project/models/deepseek-llm-7b-chat.Q4_K_M.gguf"

# Check if model exists
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load the model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=32,
    use_mlock=True,
    use_mmap=True,
    verbose=True
)

# Define the /generate endpoint
@app.post("/generate")
def generate_text(request: RequestData):
    try:
        # Enforce English response + wrap prompt
        prompt = f"You are a helpful AI assistant. Please respond in English only.\n\n{request.prompt}"

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


# Start with: python deepseek.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11436)