import os
from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path

PROJECT_ID = "proposal-auto-ai-internal"
LOCATION = "global"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

client = genai.Client(vertexai=True,
                project=PROJECT_ID,
                location=LOCATION,)

# model_name = "gemini-2.0-flash" # @param ["gemini-1.5-flash-latest","gemini-2.0-flash-lite-preview-02-05","gemini-2.0-flash","gemini-2.0-pro-exp-02-05"] {"allow-input":true}

model_name = "gemini-2.5-flash"

import google.generativeai as genai




def extract_file_key(figma_url):
    """
    Extracts the file key from a Figma URL.
    Expected format: 
    https://www.figma.com/design/<file_key>/<...>?...
    """
    match = re.search(r"figma\.com/design/([^/?]+)", figma_url)
    if match:
        return match.group(1)
    return None



def export_figma_nodes(
    file_key: str,
    node_ids: list[str],
    access_token: str,
    format: str = "png",
    scale: float = 1.0,
    output_dir: str = "./downloads"
) -> dict:
    """
    Export selected nodes from a Figma file as SVG or PNG
     
    Args:
        file_key (str): Figma file key (found in the URL)
        node_ids (List[str]): List of node IDs to export
        access_token (str): Figma API access token
        format (str): Export format - 'png', 'svg', or 'pdf'
        scale (float): Export scale (1.0 = 1x, 2.0 = 2x, etc.)
        output_dir (str): Directory to save exported files
    
    Returns:
        dict: Results with success/failure info for each node
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare API request
    url = f"https://api.figma.com/v1/images/{file_key}"

    is_api = True
    if is_api:
        headers = {
        "Authorization": f"Bearer {access_token}"
            }
        
    params = {
        "ids": ",".join(node_ids),
        "format": format,
        "scale": scale
    }
    
    try:
        # Make API request to get download URLs
        print("Requesting export URLs from Figma API...")
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("err"):
            return {"error": data["err"], "results": {}}
        
        # Download each file
        results = {}
        images = data.get("images", {})
        
        for node_id in node_ids:
            if node_id in images and images[node_id]:
                try:
                    # Download the file
                    file_url = images[node_id]
                    file_response = requests.get(file_url)
                    file_response.raise_for_status()
                    
                    # Save the file
                    file_extension = format.lower()
                    filename = f"{node_id}.{file_extension}"
                    filepath = os.path.join(output_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(file_response.content)
                    
                    results[node_id] = {
                        "success": True,
                        "filepath": filepath,
                        "url": file_url
                    }
                    print(f"✓ Downloaded: {filename}")
                    
                except Exception as e:
                    results[node_id] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"✗ Failed to download {node_id}: {e}")
            else:
                results[node_id] = {
                    "success": False,
                    "error": "No download URL provided by Figma API"
                }
                print(f"✗ No URL for node {node_id}")
        
        return {"results": results}
        
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}", "results": {}}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "results": {}}




def check_and_resize_image(image_path):
    """
    Check image dimensions and resize if necessary.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        str: Path to the processed image (either original or resized)
    """
    
    # Open and check image dimensions
    with Image.open(image_path) as img:
        width, height = img.size
        
        # Check if resizing is needed
        if width > 1400 or height > 700:
            # Resize the image to 1250 x 650
            resized_img = img.resize((1250, 650), Image.Resampling.LANCZOS)
            
            # Create a resized file in current directory
            original_name = Path(image_path).stem
            original_ext = Path(image_path).suffix
            resized_path = f"{original_name}_resized{original_ext}"
            
            # Save the resized image
            resized_img.save(resized_path, optimize=True, quality=95)
            
            return resized_path
        else:
            # Return original path if no resizing needed
            return image_path



def gemini_llm_call(user_prompt, img_path):
  

#   Load and resize image
  img = Image.open(img_path)
  img = img.resize((800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS) # Resizing to speed-up rendering

  # Analyze the image using Gemini
  image_response = client.models.generate_content(
      model=model_name,
      contents=[
          img,
          user_prompt
      ],
      config = types.GenerateContentConfig(
          temperature=0.5
      )
  )

  # Check response
  print(f"\n\n@@@GEMINI RESPONSE:\n {image_response.text}\n\n")
  return image_response.text