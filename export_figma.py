import requests
import os
from typing import List, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
    node_ids: List[str],
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


def get_figma_file_nodes(file_key: str, access_token: str) -> dict:
    """
    Get all nodes from a Figma file to find node IDs
    
    Args:
        file_key (str): Figma file key
        access_token (str): Figma API access token
    
    Returns:
        dict: File structure with all nodes
    """
    url = f"https://api.figma.com/v1/files/{file_key}"
    is_api = True
    if is_api:
        headers = {
        "Authorization": f"Bearer {access_token}"
            }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_figma_access(file_key: str, access_token: str) -> dict:
    """
    Test if you have access to a Figma file
    
    Args:
        file_key (str): Figma file key
        access_token (str): Figma API access token
    
    Returns:
        dict: Test results
    """
    url = f"https://api.figma.com/v1/files/{file_key}"

    is_api = True
    if is_api:
        headers = {
        "Authorization": f"Bearer {access_token}"
            }
    
    try:
        print(f"Testing access to file: {file_key}")
        print(f"Using token: {access_token[:10]}...")
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Access successful!")
            print(f"File name: {data.get('name', 'Unknown')}")
            return {"success": True, "data": data}
        elif response.status_code == 403:
            print("✗ 403 Forbidden - Check:")
            print("  1. Your access token is valid")
            print("  2. You have access to this file")
            print("  3. The file is not private/restricted")
            return {"success": False, "error": "Forbidden", "status": 403}
        elif response.status_code == 404:
            print("✗ 404 Not Found - File key might be incorrect")
            return {"success": False, "error": "File not found", "status": 404}
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return {"success": False, "error": response.text, "status": response.status_code}
            
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return {"success": False, "error": str(e)}



# Example usage
if __name__ == "__main__":
    # Your Figma API credentials
    FIGMA_URL = "https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=204-6562&t=dMnGxshxxozW66yo-4"
    ACCESS_TOKEN = "figu_fEgEwJ5kyk8h_R0_rNezor4j4BH80difoFXauIFk"
    FILE_KEY = extract_file_key(figma_url=FIGMA_URL)  # Found in Figma URL

    # First, test access to your file
    print("=== Testing Figma Access ===")
    test_result = test_figma_access(FILE_KEY, ACCESS_TOKEN)
    
    if test_result["success"]:
        # Node IDs you want to export (you can get these from get_figma_file_nodes)
        NODE_IDS = ["I288:8218;12625:122802"]
        
        # Export as SVG
        results = export_figma_nodes(
            file_key=FILE_KEY,
            node_ids=["I288:8218;12625:122802"],
            access_token=ACCESS_TOKEN,
            format="png",
            scale=1.0,
            output_dir="./figma_exports"
        )
        
        print("Export results:", results)