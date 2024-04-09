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
    timeout = httpx.Timeout(80.0, read=80.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(LLM_API_URL, json={"text": text})
            response.raise_for_status()  # Will raise an exception for HTTP error responses
            response_data = response.json()
    except Exception as e:
        return templates.TemplateResponse("chat_form.html", {"request": request, "response": f"Error: {e}"})

    llm_response = response_data.get("result", "No response from LLM server.")
    return templates.TemplateResponse("chat_form.html", {"request": request, "response": llm_response})

if __name__ == "__main__":
    uvicorn.run("webfront:app", host="0.0.0.0", port=WEB_SERVER_PORT, reload=True)
