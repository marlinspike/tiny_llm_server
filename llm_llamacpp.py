import os
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import logging
from llama_cpp import Llama
import torch
from huggingface_hub import hf_hub_download

load_dotenv()
OUTPUT_TOKENS = int(os.getenv("OUTPUT_TOKENS", 1000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
LLM_SERVER_PORT = int(os.getenv("LLM_SERVER_PORT", 6001))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
LOG_QUERIES = os.getenv("LOG_QUERIES", "false")

logging.basicConfig(filename='app.log', level=LOG_LEVEL.upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

models = [
    {"friendly_name": "mistral", "short_name": "Mistral-7B-v0.1-GGUF", "vendor_name": "TheBloke", "filename": "mistral-7b-v0.1.Q4_K_M.gguf"},
]
app = FastAPI()

llm = None

class PredictRequest(BaseModel):
    text: str

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_model(model_config):
    model_identifier = f"{model_config['vendor_name']}/{model_config['short_name']}"
    model_directory = f"./models/{model_config['vendor_name']}/{model_config['short_name']}"
    model_filename = model_config["filename"]

    model_path = os.path.join(model_directory, "models--" + model_identifier.replace("/", "--"), "snapshots")
    if os.path.exists(model_path):
        # Model is already downloaded, return the path to the model file
        snapshot_folder = os.listdir(model_path)[0]
        model_file_path = os.path.join(model_path, snapshot_folder, model_filename)
        print(f"Model '{model_identifier}' is already downloaded.")
        return model_file_path
    
    # Model is not downloaded, download it
    if not os.path.exists(os.path.join(model_directory, model_filename)):
        print(f"Downloading model '{model_identifier}' to '{model_directory}'...")
        ensure_path_exists(model_directory)

        hf_hub_download(repo_id=model_identifier, filename=model_filename, cache_dir=model_directory, force_download=True)
        print("Model downloaded and saved successfully.")
    
    return os.path.join(model_directory, model_filename)

async def startup_event():
    global llm
    parser = argparse.ArgumentParser(description="FastAPI model serving application.")
    parser.add_argument("-m", "--model", type=str, help="Model to use by friendly name", default="tinyllama")
    parser.add_argument("-d", "--download", action="store_true", help="Download the model")
    args, unknown = parser.parse_known_args()

    selected_model_config = next((m for m in models if m["friendly_name"] == args.model), None)
    if not selected_model_config:
        available_models = ", ".join([m["friendly_name"] for m in models])
        print(f"Model '{args.model}' not found. Available models: {available_models}")
        print("Please select one of the available models by using the -m option.")
        exit(1)

    model_path = download_model(selected_model_config) if args.download else download_model(selected_model_config)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please make sure the model file is downloaded and available at the specified location.")
        exit(1)

    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    n_gpu_layers = 1 if device.type in ('cuda', 'mps') else 0
    print(f"Using device: {device}")
    print(f"Number of GPU layers: {n_gpu_layers}")

    # Set n_threads to a positive value (e.g., 8)
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=2048, n_batch=512, n_threads=8)


async def shutdown_event():
    global llm
    llm = None

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

@app.post("/predict")
async def predict(request: PredictRequest):
    global llm
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded correctly")

    if (LOG_QUERIES == "true"):
        logging.info(f"LLM Prompt: {request.text}")

    result = llm(
        request.text, 
        max_tokens=OUTPUT_TOKENS,
        stop=["</s>"],
        temperature=TEMPERATURE
    )

    return {"result": result["choices"][0]["text"]}

def run_server():
    config = uvicorn.Config("llm_llamacpp:app", host="0.0.0.0", port=LLM_SERVER_PORT, log_level=LOG_LEVEL)
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("Server stopped.")

if __name__ == "__main__":
    run_server()