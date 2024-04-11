import os
import argparse
import torch  # Import torch to manage device allocation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import uvicorn
from dotenv import load_dotenv
import logging

load_dotenv()
OUTPUT_TOKENS = int(os.getenv("OUTPUT_TOKENS", 1000))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
LLM_SERVER_PORT = int(os.getenv("LLM_SERVER_PORT", 6001))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")
LOG_QUERIES = os.getenv("LOG_QUERIES", "false")

# Configure logging
logging.basicConfig(filename='app.log', level=LOG_LEVEL.upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Define your models configuration
models = [
    {"friendly_name": "orca", "short_name": "Orca-2-7b", "vendor_name": "microsoft"},
    {"friendly_name": "tinyllama", "short_name": "TinyLlama-1.1B-Chat-v1.0", "vendor_name": "TinyLlama"},
    {"friendly_name": "phi2", "short_name": "phi-2", "vendor_name": "microsoft"},
    {"friendly_name": "mistral", "short_name": "Mistral-7B-Instruct-v0.2", "vendor_name": "mistralai"},
]

# Setup FastAPI app
app = FastAPI()


# Determine the best available device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Explicitly set the device based on availability
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# Placeholder for the pipeline object
pipe = None

class PredictRequest(BaseModel):
    text: str

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def model_already_downloaded(model_directory):
    """Check if the model and tokenizer have been already downloaded."""
    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"]
    return all(os.path.exists(os.path.join(model_directory, file)) for file in required_files)


def download_model(model_config):
    model_identifier = f"{model_config['vendor_name']}/{model_config['short_name']}"
    model_directory = f"./models/{model_config['short_name']}"
    
    # Only check for the existence of the directory, not its contents.
    if not os.path.exists(model_directory):
        print(f"Downloading model '{model_identifier}' to '{model_directory}'...")
        ensure_path_exists(model_directory)

        tokenizer = AutoTokenizer.from_pretrained(model_identifier, cache_dir=model_directory)
        tokenizer.padding_side = 'left'  # Adjust padding side to left

        model = AutoModelForCausalLM.from_pretrained(model_identifier, cache_dir=model_directory)
        tokenizer.save_pretrained(model_directory)
        model.save_pretrained(model_directory)
        print("Model downloaded and saved successfully.")
    else:
        print(f"Model '{model_identifier}' is already downloaded.")


def load_model_from_disk(model_config):
    global pipe
    model_directory = f"./models/{model_config['short_name']}"
    print(f"Attempting to load model from {model_directory}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        tokenizer.padding_side = 'left'  # Ensure tokenizer uses left padding

        config = AutoConfig.from_pretrained(model_directory)
        model = AutoModelForCausalLM.from_pretrained(model_directory, config=config).to(device)
        
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)
        print(f"Model '{model_config['short_name']}' loaded successfully.")
    except Exception as e:
        print(f"Failed to load model '{model_config['short_name']}' from disk. Error: {e}")
        print("Please delete the model directory and try downloading again.")


async def startup_event():
    parser = argparse.ArgumentParser(description="FastAPI model serving application.")
    parser.add_argument("-m", "--model", type=str, help="Model to use by friendly name", default="tinyllama")
    parser.add_argument("-d", "--download", action="store_true", help="Download the model from Hugging Face")
    args, unknown = parser.parse_known_args()

    selected_model_config = next((m for m in models if m["friendly_name"] == args.model), None)
    if not selected_model_config:
        available_models = ", ".join([m["friendly_name"] for m in models])
        print(f"Model '{args.model}' not found. Available models: {available_models}")
        print("Please select one of the available models by using the -m option.")
        exit(1)  # Exit the application with a non-zero status to indicate an error.

    # If the model is found, proceed with the download or load from disk.
    if args.download:
        download_model(selected_model_config)
    load_model_from_disk(selected_model_config)



async def shutdown_event():
    pipe = None

app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


@app.post("/predict")
async def predict(request: PredictRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded correctly")
    
    if (LOG_QUERIES == "true"):
        logging.info(f"LLM Prompt: {request.text}")

    # Adjust parameters using environment variables
    formatted_text = f"<s>[INST] {request.text} [/INST]</s>"
    result = pipe(formatted_text, max_length=OUTPUT_TOKENS, temperature=TEMPERATURE, truncation=True, do_sample=True, return_full_text=False)
    
    return {"result": result[0]["generated_text"]}

def run_server():
    config = uvicorn.Config("app:app", host="0.0.0.0", port=LLM_SERVER_PORT, log_level=LOG_LEVEL)
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        print("Server stopped.")

if __name__ == "__main__":
    run_server()