from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import httpx
from dotenv import load_dotenv
import os
import uvicorn

# Load environment variables
load_dotenv()

# Configuration
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:6001/predict")
WEB_SERVER_PORT = int(os.getenv("WEB_SERVER_PORT", 8000))

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("chat_form.html", {"request": request, "response": None})

@app.post("/query", response_class=HTMLResponse)
async def handle_query(request: Request, text: str = Form(...)):
    # Define a custom timeout
    timeout = httpx.Timeout(400.0, read=400.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(LLM_API_URL, json={"text": text})
            response.raise_for_status()  # Will raise an exception for HTTP error responses
            response_data = response.json()
    except Exception as e:
        return templates.TemplateResponse("chat_form.html", {"request": request, "response": f"Error: {e}"})

    llm_response = response_data.get("result", "No response from LLM server.")
    llm_response = llm_response.replace("<s>", "").replace("</s>", "")
    llm_response = llm_response.replace("[INST]", "").replace("[/INST]", "")
    
    return templates.TemplateResponse("chat_form.html", {"request": request, "response": llm_response})

def run_server():
    config = uvicorn.Config("webfront:app", host="0.0.0.0", port=WEB_SERVER_PORT, log_level="info")
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
        # Place any clean-up code here
    finally:
        print("Server stopped.")

if __name__ == "__main__":
    run_server()
