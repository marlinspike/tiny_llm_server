# FastAPI Model Serving

This FastAPI application serves as a versatile API for generating text using models from Hugging Face's Transformers library. It supports dynamic model loading based on environment variables and command-line arguments, allowing for easy switching between models such as TinyLlama, Phi-2, and Orca-2-7b.

## Features

- Load models dynamically based on .env configuration or command-line arguments.
- Serve text generation models via a simple `/predict` endpoint.
- Support for downloading and saving models to a structured directory.
- Customizable settings for model behavior (e.g., output tokens, temperature).

## Setup

### Requirements

- Python 3.10+
- FastAPI
- Uvicorn
- Pydantic
- Transformers library
- python-dotenv
- hpptx

#### Configure environment variables
Copy the `.env.example` file to `.env` and adjust the settings as needed. The following variables are available:
- DEFAULT_MODEL=tinyllama
- OUTPUT_TOKENS=2000
- PORT=6001
- TEMPERATURE=0.5
- LLM_API_URL=http://localhost:6001/predict
- WEB_SERVER_PORT=8000

#### A word on CUDA
I'm runnong on a Mac, so I'm using the CPU-only version for now. An MLX port is forthcoming. If you've got a CUDA-enabled GPU, you can install the CUDA version of PyTorch, which will speed up model inference significantly.



# For CPU-only version (default)
Do nothing, it's already installed in requirements.txt

# Or for a version with CUDA support (adjust for your CUDA version)
Replace the imports with these or just run:
```pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation

First, clone this repository to your local machine:

```bash
git clone <repository-url>
cd <repository-name>```


### Configuration
DEFAULT_MODEL=tinyllama
OUTPUT_TOKENS=1000
PORT=6001
TEMPERATURE=0.5

### Running the Application
```uvicorn app:app --reload --port 6001```

Optionally, you can download a model directly (if not already done) by executing:

```python app.py -d -m <model_name>```

### Usage
Once the server is running, you can make POST requests to the /predict endpoint to generate text. Here is an example using curl:

curl -X 'POST' \
  'http://localhost:6001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Why is the sky blue?"
}'
```


