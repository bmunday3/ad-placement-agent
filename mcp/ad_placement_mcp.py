import os
import re
import base64
import json
import ast
import asyncio
from mlflow.deployments import get_deploy_client
from typing import Dict, Any, Tuple
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, StreamingResponse
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

# Initialize MLflow deployment client
client = get_deploy_client('databricks')

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

def parse_response_from_output(agent_reasoning: list) -> Tuple[str, str]:
    """
    Parse the agent reasoning output to format it for display.
    
    Args:
        agent_reasoning: The agent reasoning list (output[:-1])
        
    Returns:
        Tuple of (formatted_reasoning, empty_string) - text_response is handled separately
    """
    
    # Format agent reasoning for display
    reasoning_text = ""
    for item in agent_reasoning:
        if item.get('type') == 'function_call':
            reasoning_text += f"<div class='cot-tool-call'>\n"
            reasoning_text += f"<div class='cot-tool-call-header'>🔧 Tool Call: {item.get('name', 'Unknown')}</div>\n"
            
            # Parse and format the query from arguments
            try:
                arguments = json.loads(item.get('arguments', '{}'))
                if 'query' in arguments:
                    reasoning_text += f"<div class='cot-query-content'><strong>Search Query:</strong> {arguments['query']}</div>\n"
                else:
                    reasoning_text += f"<div class='cot-json-content'>{item.get('arguments', '')}</div>\n"
            except json.JSONDecodeError:
                reasoning_text += f"<div class='cot-json-content'>{item.get('arguments', '')}</div>\n"
            
            reasoning_text += f"</div>\n"
            
        elif item.get('type') == 'function_call_output':
            reasoning_text += f"<div class='cot-tool-result'>\n"
            reasoning_text += f"<div class='cot-tool-result-header'>📋 Tool Result</div>\n"
            
            # Parse and format the CSV results
            try:
                output_data = json.loads(item.get('output', '{}'))
                if 'value' in output_data and output_data.get('format') == 'CSV':
                    csv_content = output_data['value']
                    reasoning_text += format_csv_results(csv_content)
                else:
                    reasoning_text += f"<div class='cot-json-content'>{item.get('output', '')}</div>\n"
            except json.JSONDecodeError:
                reasoning_text += f"<div class='cot-json-content'>{item.get('output', '')}</div>\n"
            
            reasoning_text += f"</div>\n"
    
    return reasoning_text, ""

def format_csv_results(csv_content: str) -> str:
    """
    Format CSV results from vector search in a creative and readable way.
    
    Args:
        csv_content: CSV string containing page_content and metadata
        
    Returns:
        Formatted HTML string
    """
    import csv
    from io import StringIO
    
    formatted_html = "<div class='cot-csv-results'>\n"
    
    try:
        # Use Python's csv module to properly parse the CSV
        csv_reader = csv.reader(StringIO(csv_content))
        
        # Skip header row
        next(csv_reader)
        
        for row in csv_reader:
            if len(row) >= 2:
                page_content = row[0].strip()
                metadata_str = row[1].strip()
                
                # Parse metadata
                try:
                    print(f"DEBUG - metadata_str: {metadata_str}")
                    # Use ast.literal_eval to handle Python-style single quotes
                    metadata = ast.literal_eval(metadata_str)
                    print(f"DEBUG - parsed metadata: {metadata}")
                    title = metadata.get('title', 'Unknown Title')
                    scene_number = metadata.get('scene_number', 'Unknown Scene')
                    print(f"DEBUG - title: {title}, scene_number: {scene_number}")
                except Exception as e:
                    print(f"DEBUG - Parse error: {e}")
                    title = "Unknown Title"
                    scene_number = "Unknown Scene"
                
                # Store full content for modal, truncate for display
                display_content = page_content[:200] + "..." if len(page_content) > 200 else page_content
                
                # Escape backticks and quotes for JavaScript - use full content for modal
                escaped_full_content = page_content.replace('`', '\\`').replace('"', '\\"').replace("'", "\\'")
                
                formatted_html += f"""
                <div class='cot-scene-item' data-title="{title}" data-scene-number="{scene_number}" data-full-content="{escaped_full_content}">
                    <div class='cot-scene-title'>🎬 {title} - Scene {scene_number}</div>
                    <div class='cot-scene-content'>{display_content}</div>
                </div>"""
    
    except Exception as e:
        # Fallback to simple display if CSV parsing fails
        formatted_html += f"<div class='cot-csv-error'>Error parsing results: {str(e)}</div>"
        formatted_html += f"<div class='cot-csv-raw'>{csv_content}</div>"
    
    formatted_html += "</div>\n"
    return formatted_html

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
async def get_ad_placement(prompt: str) -> Tuple[str, str]:
    """
    Generate a recommendation for advertisement placement in movies or TV shows.
    
    Args:
        prompt: User query containing details of the advertisement to be placed,
               including product description, target audience, and any specific
               requirements or preferences.
               
    Returns:
        Tuple of (text_response, agent_reasoning)
    """
    try:
        response = client.predict(
            endpoint=ENDPOINT_NAME,
            inputs={"input": [{"role": "user", "content": prompt}]}
        )

        # Debug: print response structure
        # print(f"DEBUG - Response type: {type(response)}")
        # print(f"DEBUG - Response: {response}")
        # if hasattr(response, 'output'):
        #     print(f"DEBUG - Output type: {type(response.output)}")
        #     print(f"DEBUG - Output: {response.output}")
        #     if response.output and len(response.output) > 0:
        #         print(f"DEBUG - Last output item: {response.output[-1]['content']}")
        #         print(f"DEBUG - Last output item keys: {response.output[-1].keys() if hasattr(response.output[-1], 'keys') else 'No keys method'}")

        text_response = response.output[-1]['content'][0]['text']
        agent_reasoning = response.output[:-1]
        # print("DEBUG - HERE")
    except Exception as e:
        raise Exception(f"Error getting ad placement: {str(e)}, \n\n Client {type(client)}")
    
    return text_response, agent_reasoning


    # response = w.serving_endpoints.query(
    #     name=ENDPOINT_NAME,
    #     input=[
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         },
    #     ],
    # )

    # return response.choices[0].messages.content


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
            input=[
                {
                    "role": "user",
                    "content": "Hello, this is a health check.",
                },
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
        
        text_response, agent_reasoning = await get_ad_placement(prompt)
        chain_of_thought, _ = parse_response_from_output(agent_reasoning)
        
        # Add prompt section to the beginning of chain of thought
        prompt_section = f"<div class='cot-prompt-sent'><div class='cot-prompt-sent-header'>📝 Prompt sent to agent</div><div class='cot-prompt-sent-content'>{prompt}</div></div>\n\n"
        enhanced_chain_of_thought = prompt_section + chain_of_thought
        
        return JSONResponse({
            "response": text_response,
            "chain_of_thought": enhanced_chain_of_thought,
            "full_response": text_response
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def analyze_image_endpoint(request: Request) -> JSONResponse:
    """Handle image analysis for ad placement"""
    try:
        body = await request.json()
        image_data = body.get("image_data")
        target_audience = body.get("target_audience", "").strip()
        
        if not image_data:
            return JSONResponse({"error": "Image data is required"}, status_code=400)
        
        # Convert image to text description
        image_description = await image_to_text(image_data)
        
        # Create prompt for the agent
        if target_audience:
            content_placement_prompt = f"Where should I place the following advertisement: {image_description} for the following target audience: {target_audience}"
        else:
            content_placement_prompt = f"Where should I place the following advertisement: {image_description}"
        
        # Get ad placement recommendation
        text_response, agent_reasoning = await get_ad_placement(content_placement_prompt)
        chain_of_thought, _ = parse_response_from_output(agent_reasoning)
        
        # Add image description and prompt to the beginning of chain of thought
        image_description_section = f"<div class='cot-image-description'><div class='cot-image-description-header'>🖼️ Image-to-Text Output</div><div class='cot-image-description-content'>{image_description}</div></div>\n\n"
        prompt_section = f"<div class='cot-prompt-sent'><div class='cot-prompt-sent-header'>📝 Prompt sent to agent</div><div class='cot-prompt-sent-content'>{content_placement_prompt}</div></div>\n\n"
        enhanced_chain_of_thought = image_description_section + prompt_section + chain_of_thought
        
        return JSONResponse({
            "response": text_response,
            "chain_of_thought": enhanced_chain_of_thought,
            "full_response": text_response,
            "image_description": image_description
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def stream_query_endpoint(request: Request):
    """Handle streaming ad placement queries using SSE"""
    try:
        body = await request.json()
        prompt = body.get("prompt")
        
        print(f"[STREAM] Received text query, prompt length: {len(prompt) if prompt else 0}")
        
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)
        
        async def event_generator():
            try:
                print("[STREAM] Starting event generator")
                # Send prompt as first event
                prompt_section = {
                    "type": "prompt",
                    "content": prompt
                }
                yield f"data: {json.dumps(prompt_section)}\n\n"
                await asyncio.sleep(0)  # Force flush
                print("[STREAM] Sent prompt event")
                
                # Call predict_stream
                print("[STREAM] Calling predict_stream")
                
                # Track full response for formatting
                accumulated_text = ""
                reasoning_items = []
                chunk_count = 0
                
                # Run predict_stream in executor to avoid blocking
                # Create iterator
                def _stream_chunks():
                    streaming_response = client.predict_stream(
                        endpoint=ENDPOINT_NAME,
                        inputs={"input": [{"role": "user", "content": prompt}]}
                    )
                    for chunk in streaming_response:
                        yield chunk
                
                # Process chunks asynchronously
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    chunk_iter = _stream_chunks()
                    
                    while True:
                        try:
                            # Get next chunk in thread pool to avoid blocking
                            chunk = await loop.run_in_executor(executor, lambda: next(chunk_iter, None))
                            if chunk is None:
                                break
                            
                            chunk_count += 1
                            chunk_type = chunk.get('type')
                            print(f"[STREAM] Chunk #{chunk_count}, type: {chunk_type}")
                            
                            # Handle function calls (reasoning)
                            if chunk_type == 'response.output_item.done':
                                item = chunk.get('item', {})
                                item_type = item.get('type')
                                print(f"[STREAM] Item type: {item_type}")
                                
                                if item_type == 'function_call':
                                    reasoning_items.append(item)
                                    # Format and send function call
                                    event_data = {
                                        "type": "reasoning",
                                        "subtype": "function_call",
                                        "data": item
                                    }
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    await asyncio.sleep(0)  # Force flush
                                    print("[STREAM] Sent function_call event")
                                
                                elif item_type == 'function_call_output':
                                    reasoning_items.append(item)
                                    # Format and send function call output
                                    event_data = {
                                        "type": "reasoning",
                                        "subtype": "function_call_output",
                                        "data": item
                                    }
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    await asyncio.sleep(0)  # Force flush
                                    print("[STREAM] Sent function_call_output event")
                            
                            # Handle text deltas (agent response)
                            elif chunk_type == 'response.output_text.delta':
                                delta = chunk.get('delta', '')
                                accumulated_text += delta
                                event_data = {
                                    "type": "response_delta",
                                    "delta": delta
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                                await asyncio.sleep(0)  # Force flush
                                print(f"[STREAM] Sent response_delta, length: {len(delta)}")
                        
                        except StopIteration:
                            break
                
                # Send completion event
                print(f"[STREAM] Stream complete, sent {chunk_count} chunks")
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                print(f"[STREAM] Error in event_generator: {str(e)}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "type": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        print(f"[STREAM] Error in stream_query_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

async def stream_image_endpoint(request: Request):
    """Handle streaming image analysis for ad placement using SSE"""
    try:
        body = await request.json()
        image_data = body.get("image_data")
        target_audience = body.get("target_audience", "").strip()
        
        print(f"[STREAM] Received image query, target_audience: {target_audience[:50] if target_audience else 'None'}")
        
        if not image_data:
            return JSONResponse({"error": "Image data is required"}, status_code=400)
        
        async def event_generator():
            try:
                print("[STREAM] Starting image event generator")
                # Convert image to text description
                image_description = await image_to_text(image_data)
                
                # Send image description event
                image_desc_data = {
                    "type": "image_description",
                    "content": image_description
                }
                yield f"data: {json.dumps(image_desc_data)}\n\n"
                await asyncio.sleep(0)
                print("[STREAM] Sent image description event")
                
                # Create prompt for the agent
                if target_audience:
                    content_placement_prompt = f"Where should I place the following advertisement: {image_description} for the following target audience: {target_audience}"
                else:
                    content_placement_prompt = f"Where should I place the following advertisement: {image_description}"
                
                # Send prompt event
                prompt_data = {
                    "type": "prompt",
                    "content": content_placement_prompt
                }
                yield f"data: {json.dumps(prompt_data)}\n\n"
                await asyncio.sleep(0)
                print("[STREAM] Sent prompt event")
                
                # Call predict_stream with async executor
                print("[STREAM] Calling predict_stream for image")
                
                accumulated_text = ""
                chunk_count = 0
                
                # Run predict_stream in executor to avoid blocking
                def _stream_chunks():
                    streaming_response = client.predict_stream(
                        endpoint=ENDPOINT_NAME,
                        inputs={"input": [{"role": "user", "content": content_placement_prompt}]}
                    )
                    for chunk in streaming_response:
                        yield chunk
                
                # Process chunks asynchronously
                import concurrent.futures
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    chunk_iter = _stream_chunks()
                    
                    while True:
                        try:
                            chunk = await loop.run_in_executor(executor, lambda: next(chunk_iter, None))
                            if chunk is None:
                                break
                            
                            chunk_count += 1
                            chunk_type = chunk.get('type')
                            print(f"[STREAM] Image chunk #{chunk_count}, type: {chunk_type}")
                            
                            # Handle function calls (reasoning)
                            if chunk_type == 'response.output_item.done':
                                item = chunk.get('item', {})
                                item_type = item.get('type')
                                print(f"[STREAM] Item type: {item_type}")
                                
                                if item_type == 'function_call':
                                    event_data = {
                                        "type": "reasoning",
                                        "subtype": "function_call",
                                        "data": item
                                    }
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    await asyncio.sleep(0)
                                    print("[STREAM] Sent function_call event")
                                
                                elif item_type == 'function_call_output':
                                    event_data = {
                                        "type": "reasoning",
                                        "subtype": "function_call_output",
                                        "data": item
                                    }
                                    yield f"data: {json.dumps(event_data)}\n\n"
                                    await asyncio.sleep(0)
                                    print("[STREAM] Sent function_call_output event")
                            
                            # Handle text deltas (agent response)
                            elif chunk_type == 'response.output_text.delta':
                                delta = chunk.get('delta', '')
                                accumulated_text += delta
                                event_data = {
                                    "type": "response_delta",
                                    "delta": delta
                                }
                                yield f"data: {json.dumps(event_data)}\n\n"
                                await asyncio.sleep(0)
                                print(f"[STREAM] Sent response_delta, length: {len(delta)}")
                        
                        except StopIteration:
                            break
                
                # Send completion event
                print(f"[STREAM] Image stream complete, sent {chunk_count} chunks")
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                
            except Exception as e:
                print(f"[STREAM] Error in image event_generator: {str(e)}")
                import traceback
                traceback.print_exc()
                error_data = {
                    "type": "error",
                    "message": str(e)
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"[STREAM] Error in stream_image_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
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
        Route("/stream-query", stream_query_endpoint, methods=["POST"]),
        Route("/stream-image", stream_image_endpoint, methods=["POST"]),
        Mount("/static", StaticFiles(directory="."), name="static"),
        Mount("/mcp", app=mcp.streamable_http_app())
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)