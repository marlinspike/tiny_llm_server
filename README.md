# FastAPI LLM Server and Web App Tester

There are two different implementations of an LLM Server here, each fronted by a FastAPI server, which allows you to expose a locally downloaded model via a web endpoint. The first is a *pure-python implementation using PyTorch* (app.py), and the second, is a **much faster implementation** using **Llama-Cpp-Python**. 

Why two implementations? Because for some implementations, you need to reduce the surface area of packages used (authorization issues), and for others, you just need the raw speed, at the cost of a bit more complexity, while still using well-known libraries that are OSS.

**Note:** The llama-cpp-python version uses the GGUF version of the model. You may need to use hugging-face cli to download gated versions of the model, although the one provided in the code is public.

This FastAPI application serves as a versatile API for generating text using models from Hugging Face's Transformers library, and provides you a Light SLLM Server, as well as a basic GUI front-end. It supports dynamic model loading based on environment variables and command-line arguments, allowing for easy switching between models such as Mistral, TinyLlama, Phi-2, and Orca-2-7b.

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

#### Important -- For Mac Silicon Users when using llm_llamacpp version

You will need to run teh following command to ensure that llama-cpp-python uses the Mac Silicon acceleration for Metal. This will speed up the model inference significantly. You can install the package with the following command:

```
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python==0.2.27
```


### Models Supported
The models below are supported directly, and you can add more by slight modifications in the code. Mistral is the heavier model, and while you can certainly run it on a CPU, it will be considerably slower than on a GPU.

- Mistral
- TinyLlama
- Phi-2
- Orca


#### Configure environment variables
Copy the `.env.example` file to `.env` and adjust the settings as needed. The following variables are available:
```
- DEFAULT_MODEL=mistral
- OUTPUT_TOKENS=2000
- PORT=6001
- TEMPERATURE=0.5
- LLM_API_URL=http://localhost:6001/predict
- WEB_SERVER_PORT=8000
```

#### A word on CUDA
I'm runnong on a Mac, so I'm using the CPU-only version for now. An MLX port is forthcoming. If you've got a CUDA-enabled GPU, you can install the CUDA version of PyTorch, which will speed up model inference significantly. Read more here: https://pytorch.org/get-started/locally/



# For CPU-only version (default)
Do nothing, it's already installed in requirements.txt

# Or for a version with CUDA support (adjust for your CUDA version)
Replace the imports with these or just run:
```
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
```

### Installation

First, clone this repository to your local machine.


### Running the Application
You'll need two terminals to run both the LLM Server as well as the front-end GUI.

#### Running The PyTorch LLM Server

This command will check whether the model directory is present, and if not, will download the model from HuggingFace. 
```
python llm_pytorch.py -d -m <model_name>
```

This will download the model (if it's not present), and then start the FastAPI server to expose a `/predict` endpoint at the port you confirgured in the .env file.

**Note:** This version is *slower* than the Llama-Cpp-Python version, so you should keep the number of output tokens to 250 or so, and then scale up, depending on your hardware.


#### Running the Llama-Cpp-Python LLM Server

This command will check whether the model directory is present, and if not, will download the model from HuggingFace. 
```
python llm_llamacpp.py -d -m mistral
```

This will download the model (if it's not present), and then start the FastAPI server to expose a `/predict` endpoint at the port you confirgured in the .env file.

**Note:** This version is *much faster* than the PyTorch version, so you can bump up the number of output tokens to 1,000 or more, depending on your hardware. Response time are an order of magnitude faster than the PyTorch version.


#### Running the Front-End GUI
On another terminal window, (first having started the Python Environment), run the following command to start the FastAPI server:
```python webfront.py```

### Usage
Once the LLM Server is running, you can make POST requests to the /predict endpoint to generate text, or just use the Webfront server to use the LLM from a browser. Here is an example using curl:

```
curl -X 'POST' \
  'http://localhost:6001/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Why is the sky blue?"
}'
```


