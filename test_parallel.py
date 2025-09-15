import cv2
from PIL import Image
import json
import time
import threading
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import requests
import re
import shutil
import hashlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import your actual functions
from utils import gemini_llm_call
from google_integration import gemini_image
from claude_integration import claude_image
from prompt import proper_child_prompt, heirarchy_end_prompt
from export_figma import export_figma_nodes

# Configuration
MAX_WORKERS = 8  # Reduced to avoid rate limiting
OUTPUT_DIR = "./figma_exports_appmod_s1_chrome_enhanced_parallel_new"
CROPPED_IMG_PATH = "appmod_s1_chrome_cropped.png"

# Rate limiting configuration
API_RATE_LIMIT = 0.5  # Seconds between API calls
api_call_lock = threading.Lock()
last_api_call_time = 0

# Utils
def check_and_resize_image(image_path, node_id="", additional_id="resize"):
    """
    Thread-safe function to check image dimensions and resize if necessary.
    Uses unique filenames instead of locking for thread safety.
    
    Args:
        image_path (str): Path to the input image
        node_id (str): Optional node ID for unique naming
        additional_id (str): Additional identifier for unique naming
    
    Returns:
        str: Path to the processed image (either original or resized)
    """
    try:
        # Validate input image first
        if not validate_image_file(image_path):
            safe_print(f"❌ Invalid image file for resizing: {image_path}")
            return image_path
        
        # Open and check image dimensions (no lock needed for reading)
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if resizing is needed
            if width > 1400 or height > 700:
                safe_print(f"📏 Resizing image {image_path} from {width}x{height} to 1250x650")
                
                # Create thread-safe unique filename
                original_path = Path(image_path)
                thread_id = threading.current_thread().ident
                timestamp = int(time.time() * 1000000)  # Microsecond precision
                
                # Generate safe_id using same pattern as other functions
                safe_id = hashlib.md5(f"{thread_id}_{node_id}_{additional_id}_{timestamp}".encode()).hexdigest()[:8]
                resized_filename = f"{original_path.stem}_resized_{safe_id}{original_path.suffix}"
                
                # Determine output directory
                output_dir = original_path.parent
                resized_path = output_dir / resized_filename
                
                # Ensure output directory exists
                if not ensure_directory_exists(str(output_dir)):
                    safe_print(f"❌ Could not create output directory: {output_dir}")
                    return image_path
                
                # Resize and save (file operations are atomic)
                resized_img = img.resize((1250, 650), Image.Resampling.LANCZOS)
                resized_img.save(str(resized_path), optimize=True, quality=95)
                
                safe_print(f"✅ Image resized and saved to: {resized_path}")
                return str(resized_path)
            else:
                safe_print(f"📏 Image {image_path} ({width}x{height}) doesn't need resizing")
                return image_path
                
    except Exception as e:
        safe_print(f"❌ Error in check_and_resize_image for {image_path}: {e}")
        return image_path


# Multithreading of LLM calls
def rate_limited_api_call():
    """Enforce rate limiting for API calls"""
    global last_api_call_time
    with api_call_lock:
        current_time = time.time()
        time_since_last_call = current_time - last_api_call_time
        if time_since_last_call < API_RATE_LIMIT:
            sleep_time = API_RATE_LIMIT - time_since_last_call
            time.sleep(sleep_time)
        last_api_call_time = time.time()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with api_call_lock:
        print(*args, **kwargs)
        logger.info(' '.join(str(arg) for arg in args))

def get_safe_thread_folder(node_id: str, additional_id: str = "") -> str:
    """Create a safe, unique folder name for threading"""
    thread_id = threading.current_thread().ident
    # Create a hash to ensure uniqueness and avoid invalid characters
    safe_id = hashlib.md5(f"{thread_id}_{node_id}_{additional_id}".encode()).hexdigest()[:8]
    return f"thread_{safe_id}"

def ensure_directory_exists(path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        safe_print(f"❌ Error creating directory {path}: {e}")
        return False

def validate_image_file(image_path: str) -> bool:
    """Validate that image file exists and is readable"""
    try:
        if not os.path.exists(image_path):
            safe_print(f"❌ Image file does not exist: {image_path}")
            return False
        
        # Try to open with PIL to validate
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception as e:
        safe_print(f"❌ Invalid image file {image_path}: {e}")
        return False

def calculate_relative_bounding_box(node: Dict[str, Any], frame_bbox: Dict[str, float], scale: float = 1.0, stroke_weight: float = 0.0) -> Optional[Dict[str, Any]]:
    """Calculate relative bounding box for a node relative to frame"""
    bbox = node.get('absoluteBoundingBox')
    if not bbox:
        return None
    
    stroke_adjustment = stroke_weight * scale
    return {
        'x': (bbox['x'] - frame_bbox['x']) * scale + stroke_adjustment,
        'y': (bbox['y'] - frame_bbox['y']) * scale + stroke_adjustment,
        'width': bbox['width'] * scale,
        'height': bbox['height'] * scale
    }

def draw_boxes_safe(image_path, boxes_with_colors, output_path, thickness=1, text_offset=None, show_connector=True, text_position='top'):
    """
    Thread-safe function to draw bounding boxes with colors
    
    Args:
        image_path (str): Path to input image
        boxes_with_colors (list): List of tuples [(x1, y1, x2, y2, color), ...]
                                 color can be (r, g, b) tuple or string like 'red'
        output_path (str): Path to save output image
        thickness (int): Line thickness for boxes
        text_offset (int): Text offset from box edge (auto-calculated if None)
        show_connector (bool): Whether to show connector line from text to box
        text_position (str): Position of text ('top', 'bottom', 'left', 'right', 'top-left')
    
    Returns:
        bool: True if successful, False otherwise
    
    Example:
        boxes = [
            (100, 100, 200, 200, (255, 0, 0)),  # Red box
            (300, 150, 400, 250, (0, 255, 0)),  # Green box
            (150, 300, 350, 400, (0, 0, 255))   # Blue box
        ]
        success = draw_boxes("input.png", boxes, "output.png")
    """
    
    # Thread-safe execution with matplotlib lock
    with api_call_lock:
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                print(f"❌ Image file does not exist: {image_path}")
                return False
            
            if not boxes_with_colors:
                print(f"❌ No boxes provided for drawing")
                return False
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Color mapping for string colors
            color_map = {
                'red': (1.0, 0.0, 0.0),
                'green': (0.0, 1.0, 0.0), 
                'blue': (0.0, 0.0, 1.0),
                'yellow': (1.0, 1.0, 0.0),
                'purple': (1.0, 0.0, 1.0),
                'cyan': (0.0, 1.0, 1.0),
                'orange': (1.0, 0.65, 0.0),
                'white': (1.0, 1.0, 1.0),
                'black': (0.0, 0.0, 0.0)
            }

            # Load and display image
            img = plt.imread(image_path)
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)
            
            # Draw each box with exact float coordinates
            for i, box_data in enumerate(boxes_with_colors):
                try:
                    if len(box_data) != 5:
                        print(f"⚠️ Skipping invalid box data at index {i}: {box_data}")
                        continue
                        
                    x1, y1, x2, y2, color = box_data
                    
                    # Validate coordinates
                    if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
                        print(f"⚠️ Skipping box with invalid coordinates at index {i}")
                        continue
                    
                    if x2 <= x1 or y2 <= y1:
                        print(f"⚠️ Skipping box with invalid dimensions at index {i}")
                        continue
                    
                    # Convert string color to RGB tuple if needed
                    if isinstance(color, str):
                        color = color_map.get(color.lower(), (1.0, 1.0, 1.0))
                    elif isinstance(color, tuple) and len(color) >= 3:
                        # Handle both 0-1 and 0-255 color ranges
                        if all(c > 1 for c in color[:3]):
                            # Convert BGR 0-255 to RGB 0-1
                            color = (color[2]/255.0, color[1]/255.0, color[0]/255.0)
                        else:
                            # Already in 0-1 range
                            color = color[:3]
                    else:
                        print(f"⚠️ Invalid color format at index {i}, using white")
                        color = (1.0, 1.0, 1.0)
                    
                    # Calculate width and height from float coordinates
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create rectangle patch with exact float coordinates
                    rect = patches.Rectangle((x1, y1), width, height, 
                                           linewidth=thickness, edgecolor=color, 
                                           facecolor='none', linestyle='-')
                    ax.add_patch(rect)
                    
                    # Add text label outside the bounding box with flexible positioning
                    if text_offset is None:
                        text_offset = max(15, thickness * 3)
                    
                    # Calculate text position based on text_position parameter
                    if text_position == 'top':
                        text_x, text_y = x1, y1 - text_offset
                        connector_start = (x1 + 15, y1 - text_offset + 2)
                        connector_end = (x1, y1)
                    elif text_position == 'bottom':
                        text_x, text_y = x1, y2 + text_offset
                        connector_start = (x1 + 15, y2 + text_offset - 2)
                        connector_end = (x1, y2)
                    elif text_position == 'left':
                        text_x, text_y = x1 - text_offset - 30, y1
                        connector_start = (x1 - text_offset, y1)
                        connector_end = (x1, y1)
                    elif text_position == 'right':
                        text_x, text_y = x2 + text_offset, y1
                        connector_start = (x2 + text_offset, y1)
                        connector_end = (x2, y1)
                    elif text_position == 'top-left':
                        text_x, text_y = x1 - 20, y1 - text_offset
                        connector_start = (x1 - 5, y1 - text_offset + 5)
                        connector_end = (x1, y1)
                    else:  # default to top
                        text_x, text_y = x1, y1 - text_offset
                        connector_start = (x1 + 15, y1 - text_offset + 2)
                        connector_end = (x1, y1)
                    
                    ax.text(text_x, text_y, 'BBox', color=color, fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor=color),
                            verticalalignment='bottom', weight='bold')
                    
                    # Optional: Add a small connector line from text to box
                    if show_connector:
                        ax.plot([connector_start[0], connector_end[0]], 
                                [connector_start[1], connector_end[1]], 
                                color=color, linewidth=1, alpha=0.6)
                
                except Exception as box_error:
                    print(f"⚠️ Error drawing box {i}: {box_error}")
                    continue
            
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)  # Flip y-axis to match image coordinates
            ax.axis('off')
            
            # Save with high DPI to preserve precision
            plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # Explicitly close the figure to free memory
            
            print(f"✅ Float precision boxes drawn and saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error in draw_boxes: {e}")
            # Ensure we close any open figures in case of error
            plt.close('all')
            return False

def crop_image_safe(image_path, x, y, width, height, output_path):
    """Thread-safe crop function with validation"""
    try:
        if not validate_image_file(image_path):
            return '', False
            
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not ensure_directory_exists(output_dir):
            return '', False
            
        img = cv2.imread(image_path)
        if img is None:
            safe_print(f"❌ Could not read image: {image_path}")
            return '', False

        # Validate and clamp coordinates
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = x1 + max(1, int(width)), y1 + max(1, int(height))

        img_height, img_width = img.shape[:2]
        x1 = min(x1, img_width - 1)
        y1 = min(y1, img_height - 1)
        x2 = min(x2, img_width)
        y2 = min(y2, img_height)

        if x2 <= x1 or y2 <= y1:
            safe_print(f"❌ Invalid crop dimensions: x={x}, y={y}, w={width}, h={height}")
            return '', False

        # Use simple array slicing for cropping
        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            safe_print(f"❌ Cropped image is empty")
            return '', False

        cv2.imwrite(output_path, cropped)
        safe_print(f"✅ Cropped image saved to: {output_path}")
        return output_path, True
        
    except Exception as e:
        safe_print(f"❌ Error in crop_image_safe: {e}")
        return '', False

def safe_json_parse(llm_response: str) -> Dict[str, Any]:
    """Enhanced JSON parsing with multiple strategies"""
    if not llm_response or not llm_response.strip():
        safe_print("❌ Empty LLM response")
        return {}
    
    try:
        # Strategy 1: Look for ```json blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        # Strategy 2: Look for { } blocks
        brace_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if brace_match:
            json_str = brace_match.group(0)
            return json.loads(json_str)
        
        # Strategy 3: Try to parse the entire response
        return json.loads(llm_response.strip())
        
    except json.JSONDecodeError as e:
        safe_print(f"❌ JSON parse error: {e}")
        safe_print(f"Response preview: {llm_response[:200]}...")
        return {}
    except Exception as e:
        safe_print(f"❌ Unexpected error in JSON parsing: {e}")
        return {}

def make_claude_api_call_with_retry(system_prompt: str, prompt: str, image_data_list: List[Tuple], max_retries: int = 3) -> Tuple[bool, str]:
    """Make Claude API call with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Rate limiting
            rate_limited_api_call()
            
            # Validate all image files before API call
            valid_image_data = []
            for image_path, description, metadata in image_data_list:
                if validate_image_file(image_path):
                    valid_image_data.append((image_path, description, metadata))
                else:
                    safe_print(f"⚠️ Skipping invalid image: {image_path}")
            
            if not valid_image_data:
                safe_print("❌ No valid images for API call")
                return False, ""
            
            # Make the API call
            success, response = claude_image(
                system_prompt=system_prompt,
                prompt=prompt,
                image_data_list=valid_image_data
            )
            
            if success and response and response.strip():
                return True, response
            else:
                safe_print(f"❌ API call failed or returned empty response (attempt {attempt + 1})")
                
        except Exception as e:
            safe_print(f"❌ Exception in API call (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            sleep_time = 2 ** attempt  # Exponential backoff
            safe_print(f"⏳ Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    return False, ""

def cleanup_thread_folder(thread_folder: str):
    """Safely cleanup thread folder"""
    try:
        if os.path.exists(thread_folder):
            shutil.rmtree(thread_folder)
            safe_print(f"🧹 Cleaned up thread folder: {thread_folder}")
    except Exception as cleanup_error:
        safe_print(f"⚠️ Warning: Could not clean up thread folder {thread_folder}: {cleanup_error}")

def check_if_continue_with_retry(image_path, node: Dict[str, Any], frame_bbox: Dict[str, float], 
                               scale: float = 1.0, stroke_weight: float = 0.0, max_retries=3) -> bool:
    """Enhanced check_if_continue function with better error handling"""
    node_id = node.get('id', 'unknown')
    thread_folder = get_safe_thread_folder(node_id, "continue")
    
    if not ensure_directory_exists(thread_folder):
        safe_print(f"❌ Could not create thread folder: {thread_folder}")
        return True  # Default to stop if can't create folder

    try:
        # Calculate relative bounding box
        relative_bbox = calculate_relative_bounding_box(node, frame_bbox, scale, stroke_weight)
        if not relative_bbox:
            safe_print(f"❌ Could not calculate bounding box for node {node_id}")
            return True

        # Validate bounding box dimensions
        if relative_bbox['width'] <= 0 or relative_bbox['height'] <= 0:
            safe_print(f"❌ Invalid bounding box dimensions for node {node_id}")
            return True

        safe_print(f"📦 Relative Bounding Box: {relative_bbox}")

        boxes_with_colors = [
            (
                relative_bbox['x'],
                relative_bbox['y'],
                relative_bbox['x'] + relative_bbox['width'],
                relative_bbox['y'] + relative_bbox['height'],
                (1.0, 0.0, 0.0)
            )
        ]

        comp_bb_output_path = os.path.join(thread_folder, "comp_bb_output.png")
        cropped_component_path = os.path.join(thread_folder, "cropped_component.png")
        
        # Draw bounding box
        if not draw_boxes_safe(image_path, boxes_with_colors, output_path=comp_bb_output_path, thickness=1):
            safe_print(f"❌ Failed to draw bounding boxes for node {node_id}")
            return True

        # Crop the image
        output, check = crop_image_safe(
            image_path,
            relative_bbox['x'],
            relative_bbox['y'],
            relative_bbox['width'],
            relative_bbox['height'],
            cropped_component_path
        )

        if not check:
            safe_print(f"❌ Failed to crop image for node {node_id}")
            return True

        # Get image dimensions for prompt
        try:
            comp_width, comp_height = Image.open(image_path).size
        except Exception as e:
            safe_print(f"❌ Could not get image dimensions: {e}")
            return True

        prompt = heirarchy_end_prompt(comp_width, comp_height)

        # Resize images if needed
        # resized_comp_bb_image_path = check_and_resize_image(comp_bb_output_path)
        # resized_cropped_image_path = check_and_resize_image(output)

        resized_comp_bb_image_path = check_and_resize_image(comp_bb_output_path, node_id, "comp_bb")
        resized_cropped_image_path = check_and_resize_image(output, node_id, "cropped")

        # Make API call with retry
        success, llm_response = make_claude_api_call_with_retry(
            system_prompt='',
            prompt=prompt,
            image_data_list=[
                (resized_comp_bb_image_path, "Image 1: Full UX screen with highlighted component", ""),
                (resized_cropped_image_path, "Image 2: Cropped view of component content", "")
            ]
        )

        if not success:
            safe_print(f"❌ All API call attempts failed for node {node_id}")
            return True  # Default to stop if API fails

        safe_print(f"🤖 LLM Response received: {len(llm_response)} characters")

        # Save debug info
        with api_call_lock:
            with open("appmod_s1_chrome_enhanced_parallel_new_check_if_continue.txt", "a") as f:
                f.write(f"Node ID: {node_id}\n")
                f.write(f"LLM Response: {llm_response}\n")
                f.write(f"{'='*50}\n\n")

        # Parse JSON response
        response_data = safe_json_parse(llm_response)
        
        if response_data and 'should_divide_further' in response_data:
            decision = not response_data.get('should_divide_further', True)
            safe_print(f"✅ Decision for node {node_id}: {'STOP' if decision else 'CONTINUE'}")
            return decision
        else:
            safe_print(f"❌ Could not parse valid response for node {node_id}")
            return True  # Default to stop if parsing fails
            
    except Exception as e:
        safe_print(f"❌ Error in check_if_continue for node {node_id}: {e}")
        return True
    finally:
        cleanup_thread_folder(thread_folder)

def check_if_proper_child_with_retry(image_path, parent_node: Dict[str, Any], child_node: Dict[str, Any], 
                                   frame_bbox: Dict[str, float], scale: float = 1.0, stroke_weight: float = 0.0, 
                                   max_retries=3) -> bool:
    """Enhanced check_if_proper_child function with better error handling"""
    parent_id = parent_node.get('id', 'unknown')
    child_id = child_node.get('id', 'unknown')
    thread_folder = get_safe_thread_folder(parent_id, f"child_{child_id}")

    if not ensure_directory_exists(thread_folder):
        safe_print(f"❌ Could not create thread folder: {thread_folder}")
        return True

    try:
        # Calculate bounding boxes
        parent_bbox = calculate_relative_bounding_box(parent_node, frame_bbox, scale, stroke_weight)
        child_bbox = calculate_relative_bounding_box(child_node, frame_bbox, scale, stroke_weight)

        if not parent_bbox or not child_bbox:
            safe_print(f"❌ Could not calculate bounding boxes for parent {parent_id} or child {child_id}")
            return True

        # Validate dimensions
        if parent_bbox['width'] <= 0 or parent_bbox['height'] <= 0 or child_bbox['width'] <= 0 or child_bbox['height'] <= 0:
            safe_print(f"❌ Invalid bounding box dimensions")
            return True

        safe_print(f"📦 Parent bbox: {parent_bbox}")
        safe_print(f"📦 Child bbox: {child_bbox}")

        # Crop parent region
        cropped_parent_path = os.path.join(thread_folder, "cropped_parent.png")
        output, check = crop_image_safe(
            image_path,
            parent_bbox['x'],
            parent_bbox['y'],
            parent_bbox['width'],
            parent_bbox['height'],
            cropped_parent_path
        )

        if not check:
            safe_print(f"❌ Failed to crop parent image")
            return True

        # Calculate child position relative to parent
        child_rel_x = child_bbox['x'] - parent_bbox['x']
        child_rel_y = child_bbox['y'] - parent_bbox['y']

        # Validate relative position
        if child_rel_x < 0 or child_rel_y < 0:
            safe_print(f"⚠️ Child position outside parent bounds")
            return True

        # Draw bounding boxes
        boxes_with_colors = [
            (0, 0, parent_bbox['width'], parent_bbox['height'], (1.0, 0.0, 0.0)),
            (child_rel_x, child_rel_y, child_rel_x + child_bbox['width'], child_rel_y + child_bbox['height'], (0.0, 0.0, 1.0))
        ]
        
        cropped_with_child_box_path = os.path.join(thread_folder, "cropped_parent_with_child.png")
        
        if not draw_boxes_safe(output, boxes_with_colors, output_path=cropped_with_child_box_path, thickness=1):
            safe_print(f"❌ Failed to draw parent-child boxes")
            return True

        prompt = proper_child_prompt(
            parent_width=parent_bbox['width'],
            parent_height=parent_bbox['height'],
            child_width=child_bbox['width'],
            child_height=child_bbox['height']
        )

        # Resize image if needed
        resized_cropped_with_child_box_path = check_and_resize_image(cropped_with_child_box_path, parent_id, f"child_{child_id}")

        # Make API call with retry
        success, llm_response = make_claude_api_call_with_retry(
            system_prompt="",
            prompt=prompt,
            image_data_list=[(resized_cropped_with_child_box_path, "Parent-child relationship image", "")]
        )

        if not success:
            safe_print(f"❌ All API call attempts failed for parent-child check")
            return True

        # Save debug info
        with api_call_lock:
            with open("appmod_s1_chrome_enhanced_parallel_new_parent_child_debug.txt", "a") as f:
                f.write(f"Parent ID: {parent_id} --> Child ID: {child_id}\n")
                f.write(f"LLM Response: {llm_response}\n")
                f.write(f"{'='*50}\n\n")

        # Parse response
        response_data = safe_json_parse(llm_response)
        
        if response_data and 'is_proper_child' in response_data:
            is_proper_child = response_data.get('is_proper_child', True)
            explanation = response_data.get('explanation', '')
            
            safe_print(f"✅ Parent-child decision: {'PROPER' if is_proper_child else 'WRAPPER'}")
            safe_print(f"📝 Explanation: {explanation[:100]}...")
            
            return is_proper_child
        else:
            safe_print(f"❌ Could not parse parent-child response")
            return True

    except Exception as e:
        safe_print(f"❌ Error in check_if_proper_child: {e}")
        return True
    finally:
        cleanup_thread_folder(thread_folder)

def get_visible_children_with_bbox(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get all visible children that have bounding boxes"""
    children = node.get('children', [])
    visible_children = []
    
    for child in children:
        if child.get('visible', True) and child.get('absoluteBoundingBox'):
            visible_children.append(child)
    
    return visible_children

def create_result_node(node: Dict[str, Any], frame_bbox: Dict[str, Any], scale: float = 1.0, stroke_weight: float = 0.0) -> Dict[str, Any]:
    """Create a simplified node for the result JSON"""
    result_node = {
        'id': node.get('id'),
        'name': node.get('name'),
        'type': node.get('type'),
        'children': []
    }
    
    if frame_bbox and node.get('absoluteBoundingBox'):
        relative_bbox = calculate_relative_bounding_box(node, frame_bbox, scale, stroke_weight)
        if relative_bbox:
            result_node['relativeBoundingBox'] = relative_bbox
    return result_node

class FigmaParallelProcessor:
    def __init__(self, max_workers, output_dir=OUTPUT_DIR):
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.processing_stats = {
            'nodes_processed': 0,
            'errors': 0,
            'api_calls': 0,
            'successful_decisions': 0,
            'failed_decisions': 0
        }
    
    def process_single_node_decision(self, node_data: Tuple) -> Tuple[str, bool, List]:
        """Process a single node to determine if we should continue traversal"""
        current_node, current_result, root_bbox, scale, stroke_weight, image_path = node_data
        node_id = current_node.get('id', 'unknown')
        
        try:
            safe_print(f"🔄 [Thread] Processing node: {current_node.get('name')} ({node_id})")
            self.processing_stats['nodes_processed'] += 1
            
            # Check if we should continue
            flag = check_if_continue_with_retry(image_path, current_node, root_bbox, scale, stroke_weight)
            
            if flag:
                safe_print(f"🛑 [Thread] STOPPING at node {node_id}")
                self.processing_stats['successful_decisions'] += 1
                return (node_id, True, [])
            
            # Get visible children
            visible_children = get_visible_children_with_bbox(current_node)
            children_to_process = []
            
            if len(visible_children) == 1:
                # Single child - check if it's a proper child or wrapper
                child = visible_children[0]
                proper_child = check_if_proper_child_with_retry(
                    image_path, current_node, child, root_bbox, scale, stroke_weight
                )
                
                if proper_child:
                    safe_print(f"✅ [Thread] Proper child detected for {node_id}")
                    child_result = create_result_node(child, root_bbox, scale, stroke_weight)
                    children_to_process.append((child, child_result))
                else:
                    safe_print(f"🔄 [Thread] Wrapper detected for {node_id}, unwrapping...")
                    # Handle wrapper elimination
                    node_to_check = current_node
                    while True:
                        visible_descendants = get_visible_children_with_bbox(node_to_check)
                        if not visible_descendants:
                            break
                        elif len(visible_descendants) > 1:
                            for desc in visible_descendants:
                                desc_result = create_result_node(desc, root_bbox, scale, stroke_weight)
                                children_to_process.append((desc, desc_result))
                            break
                        else:
                            node_to_check = visible_descendants[0]
            else:
                # Multiple children
                safe_print(f"⬇️ [Thread] Multiple children ({len(visible_children)}) for {node_id}")
                for child in visible_children:
                    child_result = create_result_node(child, root_bbox, scale, stroke_weight)
                    children_to_process.append((child, child_result))
            
            self.processing_stats['successful_decisions'] += 1
            return (node_id, False, children_to_process)
        
        except Exception as e:
            safe_print(f"❌ [Thread] Error processing node {node_id}: {e}")
            self.processing_stats['errors'] += 1
            self.processing_stats['failed_decisions'] += 1
            return (node_id, True, [])
    
    def parallel_hybrid_figma_traversal(self, root_node: Dict[str, Any], scale: float = 1.0, 
                                       stroke_weight: float = 0.0, image_path: str = CROPPED_IMG_PATH) -> Dict[str, Any]:
        """Parallel version of hybrid_figma_traversal with enhanced error handling"""
        
        if image_path is None or not validate_image_file(image_path):
            image_path = "appmod_s1.png"
            safe_print(f"⚠️ Cropped image not found, using original: {image_path}")

        # Validate inputs
        root_bbox = root_node.get('absoluteBoundingBox')
        if not root_bbox:
            raise ValueError("Root node must have absoluteBoundingBox")
        
        if not validate_image_file(image_path):
            raise FileNotFoundError(f"Invalid image file: {image_path}")
        
        safe_print(f"🚀 Starting Enhanced Parallel Figma Traversal")
        safe_print(f"📊 Workers: {self.max_workers}, Scale: {scale}, Stroke: {stroke_weight}")
        
        result_root = create_result_node(root_node, root_bbox, scale, stroke_weight)
        current_level = [(root_node, result_root, None)]
        level = 0
        
        while current_level:
            safe_print(f"\n{'='*60}")
            safe_print(f"📊 Level {level}: Processing {len(current_level)} nodes")
            safe_print(f"{'='*60}")
            
            next_level = []
            
            # Prepare data for parallel processing
            processing_data = []
            for current_node, current_result, parent_result in current_level:
                processing_data.append((current_node, current_result, root_bbox, scale, stroke_weight, image_path))
            
            # Process nodes in parallel with enhanced error handling
            completed_futures = 0
            total_futures = len(processing_data)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_data = {
                    executor.submit(self.process_single_node_decision, data): data 
                    for data in processing_data
                }
                
                for future in as_completed(future_to_data):
                    completed_futures += 1
                    original_data = future_to_data[future]
                    current_node = original_data[0]
                    current_result = original_data[1]
                    
                    try:
                        node_id, should_stop, children_to_process = future.result(timeout=60)  # 60 second timeout
                        
                        if should_stop:
                            current_result['children'] = []
                            safe_print(f"✅ [{completed_futures}/{total_futures}] Node {node_id} STOPPED")
                        else:
                            current_result['children'] = []
                            for child_node, child_result in children_to_process:
                                current_result['children'].append(child_result)
                                next_level.append((child_node, child_result, current_result))
                            
                            safe_print(f"✅ [{completed_futures}/{total_futures}] Node {node_id} CONTINUES ({len(children_to_process)} children)")
                    
                    except TimeoutError:
                        safe_print(f"⏰ [{completed_futures}/{total_futures}] Timeout processing node {current_node.get('id')}")
                        current_result['children'] = []
                        self.processing_stats['errors'] += 1
                    except Exception as e:
                        safe_print(f"❌ [{completed_futures}/{total_futures}] Error processing node {current_node.get('id')}: {e}")
                        current_result['children'] = []
                        self.processing_stats['errors'] += 1
            
            # Move to next level
            current_level = next_level
            level += 1
            
            safe_print(f"📊 Level {level-1} completed. Next level: {len(next_level)} nodes")
            
            # Progress summary
            if level > 1:
                safe_print(f"📈 Progress: {self.processing_stats['nodes_processed']} nodes processed, {self.processing_stats['errors']} errors")
        
        # Final statistics
        safe_print(f"\n{'='*60}")
        safe_print("🎉 Enhanced Parallel Traversal Complete!")
        safe_print(f"📊 Statistics:")
        safe_print(f"   - Total levels: {level}")
        safe_print(f"   - Nodes processed: {self.processing_stats['nodes_processed']}")
        safe_print(f"   - Successful decisions: {self.processing_stats['successful_decisions']}")
        safe_print(f"   - Failed decisions: {self.processing_stats['failed_decisions']}")
        safe_print(f"   - Errors: {self.processing_stats['errors']}")
        safe_print(f"   - Success rate: {(self.processing_stats['successful_decisions'] / max(1, self.processing_stats['nodes_processed'])) * 100:.1f}%")
        safe_print(f"{'='*60}")
        
        return result_root

def parallel_process_figma_screen(figma_json: Dict[str, Any], scale: float = 1.0, stroke_weight: float = 0.0, max_workers: int = 8) -> Dict[str, Any]:
    """Main function to process Figma screen JSON with enhanced parallel processing"""
    
    safe_print("="*80)
    safe_print("🚀 Starting Enhanced Parallel Figma JSON Processing")
    safe_print("="*80)
    
    try:
        start_time = time.time()
        processor = FigmaParallelProcessor(max_workers=max_workers)
        result = processor.parallel_hybrid_figma_traversal(figma_json, scale, stroke_weight, CROPPED_IMG_PATH)
        end_time = time.time()
        
        safe_print("="*80)
        safe_print("✅ Enhanced Parallel Processing Complete!")
        safe_print(f"📊 Result: {result.get('name')} ({result.get('type')})")
        safe_print(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
        safe_print(f"📈 Performance: {processor.processing_stats['nodes_processed'] / (end_time - start_time):.2f} nodes/sec")
        safe_print("="*80)
        
        return result
        
    except Exception as e:
        safe_print(f"❌ Critical error during processing: {str(e)}")
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        raise

# Utility functions 
def extract_file_key(figma_url):
    """Extracts the file key from a Figma URL"""
    match = re.search(r"figma\.com/design/([^/?]+)", figma_url)
    if match:
        return match.group(1)
    return None

def get_bounding_boxes_relative_to_frame(node, frame_bbox, scale=1.0, stroke_weight=0.0):
    """Returns their bounding boxes relative to the given frame"""
    results = []
    children = node.get('children', [])
    node_type = node.get('type')
    bbox = node.get('absoluteBoundingBox')
    visible = node.get('visible', True)

    if bbox and visible:
        relative_bbox = calculate_relative_bounding_box(node, frame_bbox, scale, stroke_weight)
        if relative_bbox:
            results.append({
                'id': node.get('id'),
                'name': node.get('name'),
                'type': node_type,
                'relativeBoundingBox': relative_bbox
            })

    if visible:
        for child in children:
            results.extend(get_bounding_boxes_relative_to_frame(child, frame_bbox, scale, stroke_weight))

    return results

def draw_bounding_boxes(image_path, bounding_boxes, output_path, color=(0, 255, 0), thickness=1):
    """Draws bounding boxes on the image and saves the result"""
    try:
        if not validate_image_file(image_path):
            raise FileNotFoundError(f"Invalid image file: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        for bbox_item in bounding_boxes:
            if 'relativeBoundingBox' not in bbox_item:
                continue
                
            bbox = bbox_item['relativeBoundingBox']
            x0 = int(bbox['x'])
            y0 = int(bbox['y'])
            x1 = int(bbox['x'] + bbox['width'])
            y1 = int(bbox['y'] + bbox['height'])
            
            # Validate coordinates
            if x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0:
                cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
                

        cv2.imwrite(output_path, img)
        safe_print(f"✅ Bounding boxes drawn and saved to {output_path}")
        
    except Exception as e:
        safe_print(f"❌ Error drawing bounding boxes: {e}")
        raise


# FUNCTIONS FOR DOWLOADING AND EXPORTING IMAGES PARALLELY
def collect_nodes_for_export(node):
    """Recursively collect all nodes in the tree for export."""
    nodes = [node]
    for child in node.get('children', []):
        nodes.extend(collect_nodes_for_export(child))
    return nodes

def find_node_in_tree(tree, target_id):
    """Recursively find a node by ID in the JSON tree"""
    if tree.get('id') == target_id:
        return tree
    
    for child in tree.get('children', []):
        result = find_node_in_tree(child, target_id)
        if result:
            return result
    return None

def find_parent_node(tree, target_id, parent=None):
    """Find the parent of a node with target_id in the JSON tree"""
    if tree.get('id') == target_id:
        return parent
    
    for child in tree.get('children', []):
        result = find_parent_node(child, target_id, tree)
        if result:
            return result
    return None

def get_node_fill_color(node: dict, complete_tree: dict = None) -> tuple:
    fills = node.get("fills", [])
    type = node.get("type", "")
    
    if type not in ["VECTOR",
        "LINE",
        "ELLIPSE",
        "POLYGON",
        "STAR",
        "RECTANGLE",
        "BOOLEAN_OPERATION"]:
        # Check all fills in current node first
        for fill in fills:
            if fill.get("type") == "SOLID" and "color" in fill:
                color = fill["color"]
                r = int(color.get("r") * 255)
                g = int(color.get("g") * 255)
                b = int(color.get("b") * 255)
                print(f"Found solid fill color in node {node.get('id')}: ({r}, {g}, {b})")
                return (r, g, b)
    
    # Not in ["VECTOR",
        # "LINE",
        # "ELLIPSE",
        # "POLYGON",
        # "STAR",
        # "RECTANGLE",
        # "BOOLEAN_OPERATION"]:

    # If no solid color fill found in current node, look up parent hierarchy
    if complete_tree and node.get('id'):
        parent_node = find_parent_node(complete_tree, node.get('id'))
        if parent_node:
            # Recursive call will check parent, then grandparent, etc.
            print(f"Checking parent node {parent_node.get('id')} for fill color")
            color = get_node_fill_color(parent_node, complete_tree)
            print(f"Found color in parent node {parent_node.get('id')}: {color}")
            return color
    
    # Default white if no fill found in entire node hierarchy
    print(f"No solid fill color found for node {node.get('id')}, returning default white")
    return (255, 255, 255)

def fix_transparent_background(image_path: str, bg_color=(255, 255, 255)) -> str:
    try:
        img = Image.open(image_path)
        if img.mode == "RGBA":
            alpha = img.getchannel("A")
            if any(pixel < 255 for pixel in alpha.getdata()):
                color_bg = Image.new("RGB", img.size, bg_color)
                color_bg.paste(img, mask=alpha)
                color_bg.save(image_path)
    except Exception as e:
        print(f"Error processing image: {e}")
    return image_path

def process_single_node_export(node, complete_tree, file_key, access_token, output_dir):
   node_id = node.get('id', '')
   node_name = node.get('name', 'Unnamed')
   node_type = node.get('type', 'Unknown')
   try:
       colour = get_node_fill_color(node, complete_tree)
       print(f"Processing node: {node_name} ({node_id}) - Type: {node_type}, Color: {colour} in multithreaded export")
       export_figma_nodes(
           file_key=file_key,
           node_ids=[node_id],
           access_token=access_token,
           format="png",
           scale=1.0,
           output_dir=output_dir
       )
       
       # Original image path from Figma export
       original_image_path = f"{output_dir}/{node_id}.png"
       
       # Sanitize node_id by replacing special characters
       sanitized_node_id = node_id
       for char in '<>:"/\\|?*;':
           sanitized_node_id = sanitized_node_id.replace(char, '_')
       sanitized_node_id = sanitized_node_id.strip('. ') or "unnamed"
       
       sanitized_image_path = f"{output_dir}/{sanitized_node_id}.png"
       
       if os.path.exists(original_image_path):
           # Rename to sanitized filename if different
           if original_image_path != sanitized_image_path:
               os.rename(original_image_path, sanitized_image_path)
           
           fix_transparent_background(sanitized_image_path, colour)
           print(f"Downloaded and sanitized: {node_name} ({sanitized_node_id})")
       
   except Exception as e:
       print(f"Error downloading {node_id}: {e}")

def process_node(node, complete_tree, file_key=None, access_token=None, output_dir=None):
    """
    Multithreaded export of all nodes in the tree.
    """
    file_key = file_key
    access_token = access_token
    output_dir = output_dir or OUTPUT_DIR

    if not file_key or not access_token:
        print("Missing file_key or access_token for Figma export.")
        return

    all_nodes = collect_nodes_for_export(node)
    total_nodes = len(all_nodes)
    batch_size = max(1, total_nodes // MAX_WORKERS)
    print(f"Total nodes to process: {total_nodes}, Batch size: {batch_size}, Max workers: {MAX_WORKERS}")

    for i in range(0, total_nodes, batch_size):
        batch = all_nodes[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_single_node_export,
                    n, complete_tree, file_key, access_token, output_dir
                ) for n in batch
            ]
            for future in as_completed(futures):
                try:
                    print(f"Processing future: {future.result()}")
                    future.result()
                except Exception as exc:
                    print(f"Thread raised an exception: {exc}")


# Remove Chrome Bar

def process_frame_elements(frame_node):
    """Find the exact chrome bar using precise and restrictive detection strategies."""
    try:
        # Get export scale
        export_scale = 1.0
        export_settings = frame_node.get('exportSettings', [])
        for setting in export_settings:
            if setting.get('format') == 'PNG':
                constraint = setting.get('constraint', {})
                if constraint.get('type') == 'SCALE':
                    export_scale = constraint.get('value', 1.0)
                    break
        
        frame_bbox = frame_node.get('absoluteBoundingBox')
        if not frame_bbox:
            print(f"Error: Frame node missing absoluteBoundingBox")
            return 1.0, []
        
        def has_traffic_lights_precise(node, children_nodes=[]):
            """Precisely check if node contains browser traffic light controls"""
            traffic_light_names = []
            
            # Check children for traffic light pattern - must have close/minimize/maximize
            for child in children_nodes:
                child_name = child.get('name', '').lower()
                if any(keyword in child_name for keyword in ['close', 'minimize', 'maximize', 'expand']):
                    traffic_light_names.append(child_name)
                
                # Check nested children for traffic lights
                for nested_child in child.get('children', []):
                    nested_name = nested_child.get('name', '').lower()
                    if any(keyword in nested_name for keyword in ['close', 'minimize', 'maximize', 'expand']):
                        traffic_light_names.append(nested_name)
            
            # Must have at least 2 different traffic light controls (close + minimize/maximize)
            has_close = any('close' in name for name in traffic_light_names)
            has_min_max = any(keyword in name for name in traffic_light_names for keyword in ['minimize', 'maximize', 'expand'])
            
            return has_close and has_min_max and len(traffic_light_names) >= 2
        
        def has_url_indicators_precise(node, children_nodes=[]):
            """Precisely check if node contains URL/address bar content"""
            def check_url_in_text(text_node):
                if text_node.get('type') == 'TEXT':
                    characters = text_node.get('characters', '').lower()
                    # Must contain actual URL patterns, not just domain extensions
                    strong_url_patterns = [
                        'https://', 'http://', 'www.',
                        '.com/', '.org/', '.net/', '.edu/', '.gov/',
                        'pineapp.com', 'subdomain/'
                    ]
                    return any(pattern in characters for pattern in strong_url_patterns)
                return False
            
            # Check current node and all nested children for URL content
            all_nodes = [node] + children_nodes
            for check_node in all_nodes:
                if check_url_in_text(check_node):
                    return True
                for nested_child in check_node.get('children', []):
                    if check_url_in_text(nested_child):
                        return True
                    for deep_child in nested_child.get('children', []):
                        if check_url_in_text(deep_child):
                            return True
            
            return False
        
        def has_browser_navigation_precise(node, children_nodes=[]):
            """Precisely check for browser-specific navigation elements"""
            browser_nav_keywords = [
                'back', 'forward', 'refresh', 'reload', 'home',
                'bookmark', 'favicon', 'new tab', 'browser tab'
            ]
            
            nav_elements = []
            all_nodes = [node] + children_nodes
            
            for check_node in all_nodes:
                node_name = check_node.get('name', '').lower()
                if any(keyword in node_name for keyword in browser_nav_keywords):
                    nav_elements.append(node_name)
                
                # Check children for navigation elements
                for child in check_node.get('children', []):
                    child_name = child.get('name', '').lower()
                    if any(keyword in child_name for keyword in browser_nav_keywords):
                        nav_elements.append(child_name)
            
            # Must have at least 2 browser navigation elements
            return len(nav_elements) >= 2
        
        def is_actual_chrome_position(node, frame_bbox):
            """Check if element is positioned like actual browser chrome"""
            bbox = node.get('absoluteBoundingBox')
            if not bbox:
                return False
            
            relative_y = bbox['y'] - frame_bbox['y']
            
            # Must be at the very top (within 5px) or clearly defined as browser chrome
            is_top_edge = relative_y <= 5
            
            # Chrome bars are typically wide and not too tall
            aspect_ratio = bbox['width'] / max(bbox['height'], 1)
            is_chrome_dimensions = aspect_ratio >= 8 and bbox['height'] <= 120
            
            return is_top_edge and is_chrome_dimensions
        
        def has_chrome_specific_elements(node, children_nodes=[]):
            """Look for elements that are specifically part of browser chrome"""
            chrome_specific = [
                'broswer control bar', 'browser control bar', 'chrome bar',
                'toolbar - url controls', 'tab & plus', 'browser tab',
                'url controls', 'address bar'
            ]
            
            all_nodes = [node] + children_nodes
            
            for check_node in all_nodes:
                node_name = check_node.get('name', '').lower()
                if any(specific in node_name for specific in chrome_specific):
                    return True
                
                # Check children
                for child in check_node.get('children', []):
                    child_name = child.get('name', '').lower()
                    if any(specific in child_name for specific in chrome_specific):
                        return True
            
            return False
        
        def find_precise_chrome_bar(node, depth=0):
            """
            Find chrome bar using very restrictive and precise criteria
            """
            indent = "  " * depth
            element_name = node.get('name', '').lower()
            element_type = node.get('type', '')
            element_scroll = node.get('scrollBehavior', '')
            children = node.get('children', [])
            
            
            # Restrictive detection criteria
            detection_flags = {
                'explicit_chrome_name': False,
                'sticky_scrolls': False,
                'traffic_lights_precise': False,
                'url_content_precise': False,
                'browser_navigation_precise': False,
                'chrome_specific_elements': False,
                'actual_chrome_position': False
            }
            
            # 1. Explicit chrome name detection (highest priority)
            explicit_chrome_names = ['chrome bar', 'browser bar', 'toolbar - url controls']
            detection_flags['explicit_chrome_name'] = any(name in element_name for name in explicit_chrome_names)
            
            # 2. STICKY_SCROLLS (original high-confidence indicator)
            detection_flags['sticky_scrolls'] = element_scroll == 'STICKY_SCROLLS'
            
            # 3. Precise traffic lights detection
            detection_flags['traffic_lights_precise'] = has_traffic_lights_precise(node, children)
            
            # 4. Precise URL content detection
            detection_flags['url_content_precise'] = has_url_indicators_precise(node, children)
            
            # 5. Precise browser navigation detection
            detection_flags['browser_navigation_precise'] = has_browser_navigation_precise(node, children)
            
            # 6. Chrome-specific elements detection
            detection_flags['chrome_specific_elements'] = has_chrome_specific_elements(node, children)
            
            # 7. Actual chrome positioning
            detection_flags['actual_chrome_position'] = is_actual_chrome_position(node, frame_bbox)
            
            # Count active flags
            active_flags = [flag for flag, value in detection_flags.items() if value]
            
            bbox = node.get('absoluteBoundingBox')
            
            # CHROME BAR DETECTION CRITERIA WITH MANDATORY STICKY_SCROLLS
            is_chrome_candidate = False
            
            # MANDATORY: Any element with STICKY_SCROLLS behavior must be removed
            if detection_flags['sticky_scrolls']:
                is_chrome_candidate = True
            
            # HIGH CONFIDENCE: Explicit name + position + one more criteria
            elif detection_flags['explicit_chrome_name'] and detection_flags['actual_chrome_position']:
                if any([detection_flags['traffic_lights_precise'], 
                       detection_flags['url_content_precise'],
                       detection_flags['chrome_specific_elements']]):
                    is_chrome_candidate = True
            
            # HIGHEST CONFIDENCE: Traffic lights + URL + position (complete browser chrome)
            elif (detection_flags['traffic_lights_precise'] and 
                  detection_flags['url_content_precise'] and 
                  detection_flags['actual_chrome_position']):
                is_chrome_candidate = True
            
            # Additional validation for chrome bar candidates
            if is_chrome_candidate and bbox:
                # For STICKY_SCROLLS elements, use relaxed validation since they're mandatory
                if detection_flags['sticky_scrolls']:
                    # Relaxed validation for STICKY_SCROLLS - only basic size checks
                    width_check = bbox['width'] >= 200  # Any reasonable width
                    height_check = bbox['height'] <= 200  # Allow taller sticky elements
                    position_check = True  # STICKY_SCROLLS can be anywhere
                    validation_type = "STICKY_SCROLLS (Mandatory)"
                else:
                    # Strict validation for other chrome bar candidates
                    width_check = bbox['width'] >= 400  # More restrictive width
                    height_check = 20 <= bbox['height'] <= 120  # More restrictive height range
                    
                    # Must be near the top of the frame
                    relative_y = bbox['y'] - frame_bbox['y']
                    position_check = relative_y <= 10  # Must be very close to top
                    validation_type = "Standard Chrome"
                
                if width_check and height_check and position_check:
                    
                    # Calculate confidence score with STICKY_SCROLLS bonus
                    confidence = 0
                    if detection_flags['explicit_chrome_name']:
                        confidence += 100
                    if detection_flags['sticky_scrolls']:
                        confidence += 200  # Highest confidence for STICKY_SCROLLS
                    if detection_flags['traffic_lights_precise']:
                        confidence += 60
                    if detection_flags['url_content_precise']:
                        confidence += 50
                    if detection_flags['chrome_specific_elements']:
                        confidence += 40
                    if detection_flags['actual_chrome_position']:
                        confidence += 30
                    
                    chrome_box = {
                        'id': node.get('id'),
                        'name': f"PRECISE CHROME: {node.get('name')}",
                        'type': 'browser_chrome_precise',
                        'element_type': element_type,
                        'scroll_behavior': element_scroll,
                        'detection_criteria': active_flags,
                        'confidence_score': confidence,
                        'is_mandatory': detection_flags['sticky_scrolls'],
                        'relativeBoundingBox': {
                            'x': bbox['x'] - frame_bbox['x'],
                            'y': bbox['y'] - frame_bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height']
                        }
                    }
                    return [chrome_box]
                else:
                    relative_y = bbox['y'] - frame_bbox['y'] if bbox else 'N/A'
            
            # Continue searching in children
            results = []
            for child in children:
                child_results = find_precise_chrome_bar(child, depth + 1)
                results.extend(child_results)
            
            return results

        def find_best_chrome_bar(node):
            """
            Find the most accurate Chrome Bar using precise detection.
            """
            
            chrome_results = find_precise_chrome_bar(node)
            
            if not chrome_results:
                return []
            
            # Sort by confidence score
            ranked_results = sorted(chrome_results, key=lambda x: x.get('confidence_score', 0), reverse=True)
            
            # for i, chrome in enumerate(ranked_results):
            #     criteria = chrome.get('detection_criteria', [])
            #     score = chrome.get('confidence_score', 0)
            
            best_chrome = ranked_results[0]
            # bbox = best_chrome['relativeBoundingBox']
            # criteria = best_chrome.get('detection_criteria', [])
            
            return [best_chrome]
        
        chrome_boxes = find_best_chrome_bar(frame_node)
        return export_scale, chrome_boxes
        
    except Exception as e:
        print(f"Error processing frame elements: {e}")
        return 1.0, []

def remove_chrome_bar_from_screen_precise(screen):
    """
    Remove chrome bar from screen image using precise detection and correct scaling.
    """
    chrome_box = None
    try:
        target_frame = screen
        
        # Extract strokeWeight from the main screen frame
        stroke_weight = target_frame.get('strokeWeight', 0.0)
        # Find exact chrome bar using precise detection
        export_scale, chrome_boxes = process_frame_elements(target_frame)
        
        
        chrome_box = chrome_boxes[0]
        
        # Load image
        img = cv2.imread('appmod_s1.png')
        if img is None:
            return chrome_box
        
        img_height, img_width = img.shape[:2]
        bbox_data = chrome_box['relativeBoundingBox']
        
        # Apply scale correctly - the bounding box is in Figma coordinates, image is scaled
        def apply_scale_to_bbox(bbox_data, scale, stroke_weight=0.0):
            # Account for strokeWeight as padding adjustment
            stroke_adjustment = stroke_weight * scale
            return {
                'x': int(bbox_data['x'] * scale + stroke_adjustment),
                'y': int(bbox_data['y'] * scale + stroke_adjustment),
                'width': int(bbox_data['width'] * scale),
                'height': int(bbox_data['height'] * scale)
            }
        
        scaled_bbox = apply_scale_to_bbox(bbox_data, export_scale, stroke_weight)
        
        # Calculate chrome coordinates with bounds checking
        chrome_top = max(0, scaled_bbox['y'])
        chrome_bottom = min(img_height, scaled_bbox['y'] + scaled_bbox['height'])
        
        # Validate chrome dimensions make sense
        chrome_height = chrome_bottom - chrome_top
        is_mandatory = chrome_box.get('is_mandatory', False)
        
        # Different validation for mandatory STICKY_SCROLLS vs other chrome bars
        if is_mandatory:
            # Relaxed validation for STICKY_SCROLLS elements (mandatory removal)
            if chrome_height <= 0 or chrome_height > img_height * 0.5:  # Allow up to 50% for sticky headers
                return chrome_box
            max_allowed_height = 300 * export_scale  # More generous for sticky elements
        else:
            # Strict validation for regular chrome bars
            if chrome_height <= 0 or chrome_height > img_height * 0.3:  # Chrome shouldn't be > 30% of image
                return chrome_box
            max_allowed_height = 150 * export_scale  # Stricter for regular chrome
        
        
        # Validate this is a reasonable chrome bar size
        if chrome_height > max_allowed_height:
            detection_type = "STICKY_SCROLLS" if is_mandatory else "chrome bar"
            return chrome_box
        
        # Remove chrome area
        if chrome_top <= 10:  # Chrome at top
            cropped_img = img[chrome_bottom:img_height, 0:img_width]

        else:
            # Chrome in middle - stitch parts together
            top_part = img[0:chrome_top, 0:img_width]
            bottom_part = img[chrome_bottom:img_height, 0:img_width]
            
            if top_part.shape[0] > 0 and bottom_part.shape[0] > 0:
                cropped_img = cv2.vconcat([top_part, bottom_part])
            elif bottom_part.shape[0] > 0:
                cropped_img = bottom_part
            elif top_part.shape[0] > 0:
                cropped_img = top_part
            else:
                return chrome_box
        
        # Save processed image
        if cropped_img is not None and cropped_img.size > 0:
            success = cv2.imwrite(CROPPED_IMG_PATH, cropped_img)
            if success:
                new_height, new_width = cropped_img.shape[:2]
                reduction = img_height - new_height
                reduction_percent = (reduction / img_height) * 100
            else:
                print(f"Failed to save processed image")
        
    except Exception as e:
        print(f"Error removing chrome bar for screen: {e}")
    
    return chrome_box

def remove_chrome_element_from_node(node, chrome_id):
            """Recursively remove chrome element from JSON node"""
            if node.get('id') == chrome_id:
                return True  # Signal this node should be removed
            
            children = node.get('children', [])
            if children:
                # Filter out chrome elements from children
                original_count = len(children)
                node['children'] = [child for child in children 
                                  if not remove_chrome_element_from_node(child, chrome_id)]
                removed_count = original_count - len(node['children'])
                
                if removed_count > 0:
                    print(f"Removed {removed_count} chrome element(s) from node {node.get('name', 'unnamed')}")
            
            return False

def adjust_bounding_boxes_after_chrome_removal(node, chrome_box, frame_bbox):
    """Recursively adjust bounding boxes after chrome bar removal"""
    if not chrome_box or not frame_bbox:
        return
    
    print(f"Chrome:{chrome_bar}")
    chrome_bbox = chrome_box['relativeBoundingBox']
    chrome_top = chrome_bbox['y']
    chrome_height = chrome_bbox['height']
    print(f"Adjusting bounding boxes after removing chrome bar: {chrome_height}")
    
    def adjust_node_recursive(current_node):
        bbox = current_node.get('absoluteBoundingBox')
        if bbox:
            # Calculate relative position to frame
            relative_y = bbox['y'] - frame_bbox['y']
            
            # If the node is below the chrome bar, shift it up
            if relative_y > chrome_top:
                current_node['absoluteBoundingBox']['y'] -= chrome_height
                print(f"📏 Adjusted bounding box for node {current_node.get('name', 'unnamed')}: moved up by {chrome_height}px")
        
        # Recursively adjust children
        for child in current_node.get('children', []):
            adjust_node_recursive(child)
    
    adjust_node_recursive(node)

# Enhanced main function
if __name__ == "__main__":
    try:
        start_time = time.time()

        # Figma API configuration
        figma_url = "https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=667-17232&t=zNfxGel27IgVTEwo-4"
        access_token = 'figu_fEgEwJ5kyk8h_R0_rNezor4j4BH80difoFXauIFk'
        
        file_key = extract_file_key(figma_url)
        if not file_key:
            raise ValueError("Could not extract file key from Figma URL")
        
        safe_print(f"📁 File key: {file_key}")

        # Fetch data from Figma API
        url = f'https://api.figma.com/v1/files/{file_key}'
        headers = {"Authorization": f"Bearer {access_token}"}
        
        safe_print("🌐 Fetching data from Figma API...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Save raw data
        with open('screen_data_crazy.json', 'w') as f:
            json.dump(response.json(), f, indent=2)
        
        res_json = response.json()
        safe_print(f"✅ Data fetched in {time.time() - start_time:.2f} seconds")

        # Process the data
        start_time = time.time()
        
        input_page_name = 'Baptist '
        page_nodes = res_json.get("document", {}).get('children', [])
        
        page_node = None
        for node in page_nodes:
            if node.get('name') == input_page_name:
                page_node = node
                break
        
        if not page_node:
            raise ValueError(f"Could not find page with name: {input_page_name}")

        # Find the specific frame
        frame_node = None
        for section in page_node.get('children', []):
            if section.get("name") == "Section Crazy Figma":
                safe_print("🎯 Found target section")
                for frame in section.get('children', []):
                    if frame.get('name') == 'appmod_s1':
                        frame_node = frame
                        safe_print(f"🎯 Found frame: {frame_node.get('name')} ({frame_node.get('id')})")
                        break
                break

        if not frame_node:
            raise ValueError("Could not find target frame 'appmod_s1'")

        # Remove chrome bar from the screen image
        chrome_bar = remove_chrome_bar_from_screen_precise(frame_node)

        # Removing the chrome bar node from the frame
        if chrome_bar:
            chrome_bar_nodeId = chrome_bar.get('id')
            safe_print(f"🗑️ Removing chrome bar node with ID: {chrome_bar_nodeId}")
            remove_chrome_element_from_node(frame_node, chrome_bar_nodeId)

        safe_print("📏 Adjusting bounding boxes after chrome bar removal...")
        adjust_bounding_boxes_after_chrome_removal(frame_node, chrome_bar, frame_node.get('absoluteBoundingBox'))

        # Save frame data
        frame_filename = f"{section['name'].replace(' ', '_')}_json.json"
        with open(frame_filename, "w") as f:
            json.dump(frame_node, f, indent=2)
        safe_print(f"💾 Frame data saved to {frame_filename}")



        # Extract processing parameters
        stroke_weight = frame_node.get('strokeWeight', 0.0)
        scale = 1.0
        
        export_settings = frame_node.get('exportSettings', [])
        if export_settings and 'constraint' in export_settings[0]:
            constraint = export_settings[0]['constraint']
            if constraint.get('type') == 'SCALE':
                scale = constraint.get('value', 1.0)

        safe_print(f"📊 Processing parameters: stroke_weight={stroke_weight}, scale={scale}")

        # Process with enhanced parallel processing
        result = parallel_process_figma_screen(
            frame_node,
            scale=scale,
            stroke_weight=stroke_weight,
            max_workers= MAX_WORKERS  
        )
        
        # Save results
        result_filename = "appmod_s1_chrome_enhanced_parallel_new_result.json"
        with open(result_filename, "w") as f:
            json.dump(result, f, indent=2)
        
        


        # safe_print(f"💾 Results saved to {result_filename}")
        safe_print(f"⏱️ Total processing time: {time.time() - start_time:.2f} seconds")





        start_time = time.time()
        # Parallel Image Download and Processing
        safe_print("🌐 Starting parallel image download and processing...")
        with open(f"{section['name'].replace(' ', '_')}_json.json", "r") as f:
                    frame_json = json.load(f)
        # Load the specific nodes to download
        with open(f"appmod_s1_chrome_enhanced_parallel_new_result.json", "r") as f:
            result_nodes = json.load(f)
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Process the result nodes but use frame_json for parent color lookup
        process_node(result_nodes, frame_json, file_key, access_token)
        safe_print(f"✅ Parallel image download and processing completed in {time.time() - start_time:.2f} seconds")


        # Optional: Generate visualization
        try:
            all_bb = get_bounding_boxes_relative_to_frame(
                frame_node, 
                frame_node['absoluteBoundingBox'], 
                scale=scale, 
                stroke_weight=stroke_weight
            )
            
            image_path = CROPPED_IMG_PATH
            if not os.path.exists(image_path):
                image_path = "appmod_s1.png"
            if os.path.exists(image_path ):
                draw_bounding_boxes(image_path, all_bb, "appmod_s1_chrome_enhanced_new_output_.png")
                safe_print("🎨 Visualization saved to appmod_s1_chrome_enhanced_new_output.png")
        except Exception as viz_error:
            safe_print(f"⚠️ Visualization failed: {viz_error}")

    except Exception as e:
        safe_print(f"❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise