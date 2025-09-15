import os
import re
import base64
from typing import Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
import uvicorn
# from openai import OpenAI

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Initialize FastMCP server
mcp = FastMCP("ad-placement")

# Initialize Databricks client
w = WorkspaceClient()

# Load environment variables
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME")
IMAGE_TO_TEXT_ENDPOINT_NAME = os.getenv("IMAGE_TO_TEXT_ENDPOINT_NAME")

# Set up templates
templates = Jinja2Templates(directory="templates/")

def parse_response(response: str) -> Tuple[str, str]:
    """
    Parse the response to separate chain of thought from markdown content.
    
    Args:
        response: The full response string from the agent
        
    Returns:
        Tuple of (chain_of_thought, markdown_content)
    """
    # Common patterns that might indicate the start of markdown content
    markdown_indicators = [
        r'#\s+.*',  # Headers starting with #
        r'\*\*.*\*\*',  # Bold text
        r'##\s+.*',  # Level 2 headers
        r'###\s+.*',  # Level 3 headers
        r'-\s+.*',  # List items
        r'\d+\.\s+.*',  # Numbered lists
    ]
    
    # Try to find where markdown content starts
    lines = response.split('\n')
    markdown_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if this line matches any markdown pattern
        for pattern in markdown_indicators:
            if re.match(pattern, line):
                markdown_start = i
                break
        else:
            continue
        break
    
    # Split the response
    chain_of_thought = '\n'.join(lines[:markdown_start]).strip()
    markdown_content = '\n'.join(lines[markdown_start:]).strip()
    
    # If no clear separation found, try alternative approach
    if not markdown_content:
        # Look for common markdown section headers
        section_patterns = [
            r'##\s+.*[Rr]ecommendation.*',
            r'##\s+.*[Aa]nalysis.*',
            r'##\s+.*[Ss]ummary.*',
            r'#\s+.*[Rr]ecommendation.*',
            r'#\s+.*[Aa]nalysis.*',
        ]
        
        for i, line in enumerate(lines):
            for pattern in section_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    markdown_start = i
                    chain_of_thought = '\n'.join(lines[:markdown_start]).strip()
                    markdown_content = '\n'.join(lines[markdown_start:]).strip()
                    break
            if markdown_content:
                break
    
    # Fallback: if still no separation, return the whole response as markdown
    if not chain_of_thought:
        chain_of_thought = "No chain of thought detected"
        markdown_content = response
    
    # Remove duplicate prompt content from chain of thought
    # Look for common prompt patterns that might be duplicated
    prompt_patterns = [
        r'Where should I place the following advertisement:',
        r'Describe your advertisement or content',
        r'User query:',
        r'Prompt:',
        r'I want to place an advertisement',
        r'I need to place an advertisement',
        r'Help me place an advertisement',
        r'Can you help me place',
        r'What would be the best place',
        r'Where should I advertise',
    ]
    
    # Split chain of thought into lines and filter out duplicate prompt content
    cot_lines = chain_of_thought.split('\n')
    filtered_lines = []
    
    for line in cot_lines:
        line_stripped = line.strip()
        # Skip lines that look like duplicate prompts
        is_duplicate_prompt = False
        for pattern in prompt_patterns:
            if re.search(pattern, line_stripped, re.IGNORECASE):
                is_duplicate_prompt = True
                break
        
        if not is_duplicate_prompt:
            filtered_lines.append(line)
    
    # Rejoin the filtered chain of thought
    chain_of_thought = '\n'.join(filtered_lines).strip()
    
    return chain_of_thought, markdown_content

async def image_to_text(image_data: str) -> str:
    """
    Convert image to text description using multimodal model.
    
    Args:
        image_data: Base64 encoded image data
        
    Returns:
        Text description of the image
    """
    try:        
        # Create the image data URI
        image_data_uri = f"data:image/jpeg;base64,{image_data}"  
        
        # Call the multimodal model
        response = w.serving_endpoints.query(
            name=IMAGE_TO_TEXT_ENDPOINT_NAME,     
            messages=[
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": "Describe this advertisement concisely in one sentence with a description of the brand and what is happening in the image"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_uri
                            }
                        }
                    ]
                )
            ] 
        )    
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error converting image to text: {str(e)}")

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
    response = w.serving_endpoints.query(
        name=ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.USER,
                content=prompt,
            ),
        ],
    )

    return response.choices[0].message.content


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
        response = w.serving_endpoints.query(
            name=ENDPOINT_NAME,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content="Hello, this is a health check.",
                ),
            ],
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
        chain_of_thought, markdown_content = parse_response(response)
        
        # Add prompt section to the beginning of chain of thought
        prompt_section = f"<div class='cot-prompt-sent'><div class='cot-prompt-sent-header'>📝 Prompt sent to agent</div><div class='cot-prompt-sent-content'>{prompt}</div></div>\n\n"
        enhanced_chain_of_thought = prompt_section + chain_of_thought
        
        return JSONResponse({
            "response": markdown_content,
            "chain_of_thought": enhanced_chain_of_thought,
            "full_response": response
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def analyze_image_endpoint(request: Request) -> JSONResponse:
    """Handle image analysis for ad placement"""
    try:
        body = await request.json()
        image_data = body.get("image_data")
        
        if not image_data:
            return JSONResponse({"error": "Image data is required"}, status_code=400)
        
        # Convert image to text description
        image_description = await image_to_text(image_data)
        
        # Create prompt for the agent
        content_placement_prompt = f"Where should I place the following advertisement: {image_description}"
        
        # Get ad placement recommendation
        response = await get_ad_placement(content_placement_prompt)
        chain_of_thought, markdown_content = parse_response(response)
        
        # Add image description and prompt to the beginning of chain of thought
        image_description_section = f"<div class='cot-image-description'><div class='cot-image-description-header'>🖼️ Image-to-Text Output</div><div class='cot-image-description-content'>{image_description}</div></div>\n\n"
        prompt_section = f"<div class='cot-prompt-sent'><div class='cot-prompt-sent-header'>📝 Prompt sent to agent</div><div class='cot-prompt-sent-content'>{content_placement_prompt}</div></div>\n\n"
        enhanced_chain_of_thought = image_description_section + prompt_section + chain_of_thought
        
        return JSONResponse({
            "response": markdown_content,
            "chain_of_thought": enhanced_chain_of_thought,
            "full_response": response,
            "image_description": image_description
        })
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
        Route("/analyze-image", analyze_image_endpoint, methods=["POST"]),
        Mount("/static", StaticFiles(directory="."), name="static"),
        Mount("/mcp", app=mcp.streamable_http_app())
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)