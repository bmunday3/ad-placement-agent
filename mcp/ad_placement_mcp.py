import os
import json
import requests
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn
from openai import OpenAI

# Initialize FastMCP server
mcp = FastMCP("ad-placement")

# Load environment variables
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "https://e2-demo-field-eng.cloud.databricks.com")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "agents_movie_scripts-ad_placement_agent-movie_scripts_chatbot_a")
SERVICE_PRINCIPAL_SECRET = os.getenv("SERVICE_PRINCIPAL_SECRET")

def get_databricks_token():
    """
    Return the Personal Access Token directly - no OAuth exchange needed.
    """
    if not SERVICE_PRINCIPAL_SECRET:
        print("Error: SERVICE_PRINCIPAL_SECRET environment variable not set")
        return None
    return SERVICE_PRINCIPAL_SECRET

# Initialize OpenAI client for Databricks
def get_openai_client():
    """Get OpenAI client with fresh token"""
    token = get_databricks_token()
    if not token:
        raise Exception("Failed to obtain Databricks token")
    
    return OpenAI(
        api_key=token,
        base_url=f"{DATABRICKS_HOST}/serving-endpoints"
    )

# Set up templates
templates = Jinja2Templates(directory="templates/")

@mcp.tool()
async def get_ad_placement(prompt: str) -> str:
    """
    Generate a recommendation for advertisement placement in movies or TV shows.
    
    Args:
        prompt: User query containing details of the advertisement to be placed,
               including product description, target audience, and any specific
               requirements or preferences.
               
    Returns:
        A detailed recommendation including suggested movie genres, titles,
        optimal scene types, timing, and reasoning.
    """
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=ENDPOINT_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: Failed to generate recommendation - {str(e)}"

async def home(request: Request) -> HTMLResponse:
    """Serve the HTML interface"""
    return templates.TemplateResponse("index.html", {"request": request})

async def favicon(request: Request):
    """Serve the favicon.ico request"""
    from starlette.responses import FileResponse
    import os
    svg_path = os.path.join(os.getcwd(), "databricks-symbol-color.svg")
    return FileResponse(svg_path, media_type="image/svg+xml")

async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint"""
    try:
        # Test connection to Databricks
        client = get_openai_client()
        response = client.chat.completions.create(
            model=ENDPOINT_NAME,
            messages=[{"role": "user", "content": "Hello, this is a health check."}]
        )
        return JSONResponse({"status": "healthy", "endpoint": ENDPOINT_NAME})
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)})

async def query_endpoint(request: Request) -> JSONResponse:
    """Handle ad placement queries"""
    try:
        body = await request.json()
        prompt = body.get("prompt")
        
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)
        
        response = await get_ad_placement(prompt)
        return JSONResponse({"response": response})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Create Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route("/", home),
        Route("/favicon.ico", favicon),
        Route("/health", health_check),
        Route("/query", query_endpoint, methods=["POST"]),
        Mount("/static", StaticFiles(directory="."), name="static"),
        Mount("/mcp", app=mcp.streamable_http_app())
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)