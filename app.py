import os
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Define your models configuration
models = [
    {"friendly_name": "orca", "short_name": "Orca-2-7b", "vendor_name": "microsoft"},
    {"friendly_name": "tinyllama", "short_name": "TinyLlama-1.1B-Chat-v1.0", "vendor_name": "TinyLlama"},
    {"friendly_name": "phi2", "short_name": "phi-2", "vendor_name": "microsoft"},
    {"friendly_name": "mistral", "short_name": "Mistral-7B-Instruct-v0.2", "vendor_name": "mistralai"},
]


# Setup FastAPI app
app = FastAPI()

# Placeholder for the pipeline object
pipe = None

class PredictRequest(BaseModel):
    text: str

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_model(model_config):
    model_identifier = f"{model_config['vendor_name']}/{model_config['short_name']}"
    model_directory = f"./models/{model_config['short_name']}"
    ensure_path_exists(model_directory)
    print(f"Downloading model '{model_identifier}' to '{model_directory}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_identifier, cache_dir=model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_identifier, cache_dir=model_directory)
    tokenizer.save_pretrained(model_directory)
    model.save_pretrained(model_directory)
    print("Model downloaded and saved successfully.")

def load_model_from_disk(model_config):
    global pipe
    model_directory = f"./models/{model_config['short_name']}"
    print(f"Loading model from {model_directory}...")
    config = AutoConfig.from_pretrained(model_directory)
    model = AutoModelForCausalLM.from_pretrained(model_directory, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.on_event("startup")
def startup_event():
    parser = argparse.ArgumentParser(description="FastAPI model serving application.")
    parser.add_argument("-m", "--model", type=str, help="Model to use by friendly name", default="tinyllama")
    parser.add_argument("-d", "--download", action="store_true", help="Download the model before starting the server")
    args, unknown = parser.parse_known_args()

    selected_model_config = next((m for m in models if m["friendly_name"] == args.model), None)
    if not selected_model_config:
        raise ValueError(f"Model '{args.model}' not found. Available models: " + ", ".join([m["friendly_name"] for m in models]))

    if args.download:
        download_model(selected_model_config)
    load_model_from_disk(selected_model_config)

@app.post("/predict")
async def predict(request: PredictRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded correctly")
    
    formatted_text = f"<s>[INST] {request.text} [/INST]</s>"
    result = pipe(formatted_text, max_length=500, temperature=0.7)  # Adjust parameters as needed
    
    return {"result": result[0]["generated_text"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6001)
