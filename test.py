import cv2
from PIL import Image
import json
from typing import List, Dict, Any, Optional
import requests
import re
from collections import deque
from utils import gemini_llm_call
from Test_Bounding.google_integration import gemini_image
from claude_integration import claude_image
from utils import check_and_resize_image
from prompt import proper_child_prompt, heirarchy_end_prompt

import json
import time
from collections import deque
from export_figma import export_figma_nodes
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 8
OUTPUT_DIR = "./figma_exports_appmod_s1_enhanced_parallel"


def calculate_relative_bounding_box(node: Dict[str, Any], frame_bbox: Dict[str, float], scale: float = 1.0, stroke_weight: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Calculate relative bounding box for a node relative to frame
    """
    bbox = node.get('absoluteBoundingBox')
    if not bbox:
        return None
    
    stroke_adjustment = stroke_weight * scale
    # print(f"Calculating..... stroke_adjustment: {stroke_adjustment} {scale}")
    return {
        'x': (bbox['x'] - frame_bbox['x']) * scale + stroke_adjustment,
        'y': (bbox['y'] - frame_bbox['y']) * scale + stroke_adjustment,
        'width': bbox['width'] * scale,
        'height': bbox['height'] * scale
    }


def draw_boxes(image_path, boxes_with_colors, output_path, thickness=1,text_offset=None, show_connector=True, text_position='top'):
    """
    Simple function to draw bounding boxes with colors
    
    Args:
        image_path (str): Path to input image
        boxes_with_colors (list): List of tuples [(x1, y1, x2, y2, color), ...]
                                 color can be (r, g, b) tuple or string like 'red'
        output_path (str): Path to save output image
        thickness (int): Line thickness for boxes
    
    Example:
        boxes = [
            (100, 100, 200, 200, (255, 0, 0)),  # Red box
            (300, 150, 400, 250, (0, 255, 0)),  # Green box
            (150, 300, 350, 400, (0, 0, 255))   # Blue box
        ]
        draw_boxes("input.png", boxes, "output.png")
    """
    
    # Color mapping for string colors
    # color_map = {
    #     'red': (0, 0, 255),
    #     'green': (0, 255, 0), 
    #     'blue': (255, 0, 0),
    #     'yellow': (0, 255, 255),
    #     'purple': (255, 0, 255),
    #     'cyan': (255, 255, 0),
    #     'orange': (0, 165, 255),
    #     'white': (255, 255, 255),
    #     'black': (0, 0, 0)
    # }
    
    # # Load image
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise FileNotFoundError(f"Image not found: {image_path}")
    
    # # Draw each box
    # for box_data in boxes_with_colors:
    #     x1, y1, x2, y2, color = box_data
        
    #     # Convert string color to BGR tuple if needed
    #     if isinstance(color, str):
    #         color = color_map.get(color.lower(), (255, 255, 255))
        
    #     # Draw rectangle
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    #     cv2.putText(img, 'BBox', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    # # Save result
    # cv2.imwrite(output_path, img)
    # print(f"Boxes drawn and saved to: {output_path}")
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
    for box_data in boxes_with_colors:
        x1, y1, x2, y2, color = box_data
        
        # Convert string color to RGB tuple if needed
        if isinstance(color, str):
            color = color_map.get(color.lower(), (1.0, 1.0, 1.0))
        elif isinstance(color, tuple) and all(c > 1 for c in color[:3]):
            # Convert BGR 0-255 to RGB 0-1
            color = (color[2]/255.0, color[1]/255.0, color[0]/255.0)
        
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
    
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Flip y-axis to match image coordinates
    ax.axis('off')
    
    # Save with high DPI to preserve precision
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Float precision boxes drawn and saved to: {output_path}")


def check_if_continue(image_path, node: Dict[str, Any], frame_bbox: Dict[str, float], scale: float = 1.0, stroke_weight: float = 0.0) -> bool:
    """Check if we should continue traversing based on external function"""

    # Calculate relative bounding box for the current node
    comp_width, comp_height = Image.open(image_path).size

    # crop the image with given component bounding box
    relative_bbox = calculate_relative_bounding_box(node, frame_bbox, scale, stroke_weight)
    print(f"2. Relative Bounding Box: {relative_bbox}\n\n")

    boxes_with_colors = [
        (
            relative_bbox['x'],
            relative_bbox['y'],
            relative_bbox['x'] + relative_bbox['width'],
            relative_bbox['y'] + relative_bbox['height'],
            # (0, 0, 255)
            (1.0, 0.0, 0.0)
        )
    ]
    draw_boxes(image_path, boxes_with_colors, output_path="comp_bb_output.png", thickness=1)

    # Crop the image
    output, check = crop_image(
        image_path,
        relative_bbox['x'],
        relative_bbox['y'],
        relative_bbox['width'],
        relative_bbox['height'],
        "cropped_component.png"
    )

    # # --- Draw children's bounding boxes on cropped image ---
    # children = node.get('children', [])
    # child_boxes = []
    # for child in children:
    #     if child.get('visible', True) and child.get('absoluteBoundingBox'):
    #         child_bbox = calculate_relative_bounding_box(child, frame_bbox, scale, stroke_weight)
    #         # Adjust child bbox to be relative to the cropped region
    #         adj_x = child_bbox['x'] - relative_bbox['x']
    #         adj_y = child_bbox['y'] - relative_bbox['y']
    #         child_boxes.append((
    #             adj_x,
    #             adj_y,
    #             adj_x + child_bbox['width'],
    #             adj_y + child_bbox['height'],
    #             (255, 0, 0)  # Red for children
    #         ))
    # if child_boxes:
    #     draw_boxes("cropped_component.png", child_boxes, output_path="cropped_component_with_children.png", thickness=2)
    #     cropped_image_for_llm = "cropped_component_with_children.png"
    # else:
    #     cropped_image_for_llm = "cropped_component.png"

    prompt = heirarchy_end_prompt(comp_width, comp_height)

    if check:

        ##################### GEMINI IMAGE CALL #####################
        # llm_response = gemini_image(
        #     system_prompt="",
        #     prompt=prompt,
        #     image_data_list=[
        #         ("comp_bb_output.png", "Image-1"),
        #         (output, "Image-2")
        #     ],
        # )


        ##################### CLAUDE IMAGE CALL #####################
        # First resize the image if needed
        resized_comp_bb_image_path = check_and_resize_image("comp_bb_output.png")
        resized_cropped_image_path = check_and_resize_image(output)

        _, llm_response = claude_image(system_prompt='', prompt=prompt,
                                    image_data_list=[
                                        (resized_comp_bb_image_path, "Image 1: Full UX screen with YELLOW highlighted component, with bounding box label as 'BBox', just above each bounding box. (shows context)", ""),
                                        (resized_cropped_image_path, "Image 2: Cropped view of just the component content (shows internal detail)", "")
                                    ]
                                    )

        print(f"LLM RESPONSE: {llm_response}")

        with open("appmod_s1_enhanced_parallel_check_if_continue.txt", "a") as f:
            # Append parent-child relationship info
            f.write(f"Parent id:{node.get('id')}\n")
            f.write(f"LLM Response: {llm_response}\n\n")
            f.write(f"-------------------------------------------------------\n\n")

        # extract json from ```json and ```
        try:
            start_idx = llm_response.find('```json') + len('```json')
            end_idx = llm_response.find('```', start_idx)
            json_str = llm_response[start_idx:end_idx].strip()
            response_data = json.loads(json_str)

            decision = not response_data.get('should_divide_further', True)  # True = STOP here, False = CONTINUE deeper
            return decision

        except Exception:
            print(f"❌ Error parsing LLM response: {llm_response}")
            return True
    return True  # Default to stopping if no valid response

def check_if_proper_child(image_path, parent_node: Dict[str, Any], child_node: Dict[str, Any], frame_bbox: Dict[str, float], scale: float = 1.0, stroke_weight: float = 0.0) -> bool:
    """
    Check if single child is a proper child by analyzing parent-child relationship.
    Sends cropped parent image with both parent (red) and child (blue) bounding boxes drawn inside.
    """
    comp_width, comp_height = Image.open(image_path).size

    # Calculate relative bounding boxes for both parent and child (relative to frame)
    parent_bbox = calculate_relative_bounding_box(parent_node, frame_bbox, scale, stroke_weight)
    child_bbox = calculate_relative_bounding_box(child_node, frame_bbox, scale, stroke_weight)

    print(f"Parent Relative Bounding Box: {parent_bbox}")
    print(f"Child Relative Bounding Box: {child_bbox}")

    # Crop parent region from the image
    cropped_parent_path = "cropped_parent.png"
    output, check = crop_image(
        image_path,
        parent_bbox['x'],
        parent_bbox['y'],
        parent_bbox['width'],
        parent_bbox['height'],
        cropped_parent_path
    )

    if check:
        # Calculate child bbox relative to cropped parent
        child_rel_x = child_bbox['x'] - parent_bbox['x']
        child_rel_y = child_bbox['y'] - parent_bbox['y']

        # Draw both parent (red) and child (blue) bounding boxes on cropped parent image
        boxes_with_colors = [
            # Parent box (full cropped image) in RED
            (0, 0, parent_bbox['width'], parent_bbox['height'], (1.0, 0.0, 0.0)),  # RED in BGR
            # Child box in BLUE
            (child_rel_x, child_rel_y, child_rel_x + child_bbox['width'], child_rel_y + child_bbox['height'], (0.0, 0.0, 1.0))  # BLUE in BGR
        ]
        cropped_with_child_box_path = "cropped_parent_with_child.png"
        draw_boxes(
            output,
            boxes_with_colors,
            output_path=cropped_with_child_box_path,
            thickness=1
        )


        prompt = proper_child_prompt(
            parent_width=parent_bbox['width'],
            parent_height=parent_bbox['height'],
            child_width=child_bbox['width'],
            child_height=child_bbox['height']
        )

        ###################### GEMINI IMAGE CALL #####################
        # llm_response = gemini_image(
        #     system_prompt="",
        #     prompt=prompt,
        #     image_data_list=[(cropped_with_child_box_path, "Image-1")],
        # )

        ##################### CLAUDE IMAGE CALL #####################
        # First resize the image if needed
        resized_cropped_with_child_box_path = check_and_resize_image(cropped_with_child_box_path)
        _, llm_response = claude_image(
            system_prompt="",
            prompt=prompt,
            image_data_list=[(resized_cropped_with_child_box_path, "Image-1", "")],
        )

        with open("appmod_s1_enhanced_parallel_parent_child_debug.txt", "a") as f:
            # Append parent-child relationship info
            f.write(f"Parent id:{parent_node.get('id')} --> Children id:{child_node.get('id')}\n")
            f.write(f"LLM Response: {llm_response}\n\n")
            f.write(f"-------------------------------------------------------\n\n")



        print(f"PARENT-CHILD LLM RESPONSE: {llm_response}")

        # Extract and parse JSON response properly
        try:
            start_idx = llm_response.find('```json')
            start_idx += len('```json')
            end_idx = llm_response.find('```', start_idx)
            json_str = llm_response[start_idx:end_idx].strip()
            response_data = json.loads(json_str)

            is_proper_child = response_data.get('is_proper_child', True)
            explanation = response_data.get('explanation', '')

            print(f"Is proper child: {is_proper_child}")
            print(f"Explanation: {explanation}")

            return is_proper_child  # True = proper child, False = unnecessary wrapper

        except Exception as e:
            print(f"❌ Error parsing parent-child LLM response: {llm_response}")
            print(f"Error: {e}")
            return True
    return True  # Default to True if no valid response or cropping failed


def get_visible_children_with_bbox(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get all visible children that have bounding boxes
    """
    children = node.get('children', [])
    visible_children = []
    
    for child in children:
        if child.get('visible', True) and child.get('absoluteBoundingBox'):
            visible_children.append(child)
    
    return visible_children

def create_result_node(
    node: Dict[str, Any],
    frame_bbox: Dict[str, Any],
    scale: float = 1.0,
    stroke_weight: float = 0.0
) -> Dict[str, Any]:
    """
    Create a simplified node for the result JSON.
    Appends the calculated relative bounding box if possible.
    """
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

def hybrid_figma_traversal(root_node: Dict[str, Any], scale: float = 1.0, stroke_weight: float = 0.0) -> Dict[str, Any]:
    """
    Hybrid DFS/BFS traversal of Figma JSON structure
    
    IMPORTANT BEHAVIOR:
    - When external function returns True: STOPS traversing and adds ONLY immediate children (no descendants)
    - When external function returns False: CONTINUES deeper to find and include descendants
    
    Args:
        root_node: The root node to start traversal from (e.g., "Out of Box")
        scale: Scale factor for bounding box calculations
        stroke_weight: Stroke weight adjustment
    
    Returns:
        Dict: Simplified JSON structure with nodes based on external function decisions
    """
    
    
    # Create root result node
    root_bbox = root_node.get('absoluteBoundingBox')
    if not root_bbox:
        raise ValueError("Root node must have absoluteBoundingBox")
    
    print(f"Stroke weight: {stroke_weight}, Scale: {scale}")
    result_root = create_result_node(root_node, root_bbox, scale, stroke_weight)
    
    # Queue for BFS-style processing: (current_node, result_node, frame_bbox)
    queue = deque([(root_node, result_root, root_bbox)])
    
    while queue:
        current_node, current_result, frame_bbox = queue.popleft()
        
        print(f"Processing node: {current_node.get('name')} ({current_node.get('type')} {current_node.get('id')})")
        
        # add function to send bb of current node to external function
        image_path = "appmod_s1.png"  # Path to the image to be processed
        print(f"1. Relative Bounding Box :{frame_bbox}")
        # here frame_bbox is constant for whole flow. It is basically the bounding box of whole screen.
        flag = check_if_continue(image_path,current_node, root_bbox, scale, stroke_weight)

        # Get visible children with bounding boxes
        visible_children = get_visible_children_with_bbox(current_node)
        
        if not visible_children:
            print(f"  No visible children found for {current_node.get('name')}")
            continue
        
        print(f"  Found {len(visible_children)} visible children")
        
        # Calculate relative bounding boxes for all children
        children_bboxes = []
        children_data = []
        
        for child in visible_children:
            rel_bbox = calculate_relative_bounding_box(child, root_bbox, scale, stroke_weight)
            if rel_bbox:
                children_bboxes.append(rel_bbox)
                children_data.append((child, rel_bbox))
        
        if not children_bboxes:
            print(f"  No valid bounding boxes calculated for children")
            continue
        
        
        
        # flag = external_function_mock("data_ai.png",children_bboxes, node_info)
        print(f"  External function returned: {flag}")
        
        if flag:
            # Flag is True: STOP traversing and add ONLY the current node (no children or descendants)
            print(f"  🛑 STOPPING HERE: Adding ONLY current node to result (NO children or descendants)")
            current_result['children'] = []  # Explicitly set empty - no children
            continue  # Skip further processing for this node
        else:
            # Flag is False: Continue traversing deeper into descendants
            if len(children_data) == 1:
                print(f"  🔍 SINGLE CHILD DETECTED: Analyzing parent-child relationship")
                
                child, rel_bbox = children_data[0]
                
                # Call check_if_proper_child for parent-child analysis
                proper_child = check_if_proper_child(
                    image_path, current_node, child, root_bbox, scale, stroke_weight
                )
                if proper_child:
                    print(f"  ✅ Proper child detected: Adding to result")
                    print(f"  ⬇️ CONTINUING DEEPER: Traversing into {len(children_data)} children and their descendants")
                    for child, rel_bbox in children_data:
                        # Create result node for this child
                        child_result = create_result_node(child, root_bbox, scale, stroke_weight)
                        current_result['children'].append(child_result)
                        
                        # Add to queue for further processing (to find their descendants)
                        child_frame_bbox = child.get('absoluteBoundingBox')
                        if child_frame_bbox:
                            queue.append((child, child_result, child_frame_bbox))
                else:
                    # Add logic to eleminate unnecessary wrapper and continue deeper and add descendents with parent as current_node not the child
                    print(f"  ❌ Unnecessary wrapper detected: Skipping child {child.get('name')}")
                    # Traverse deeper into descendants of current_node until you get multiple children or no children
                    node_to_check = current_node
                    while True:
                        visible_descendants = get_visible_children_with_bbox(node_to_check)
                        if not visible_descendants:
                            # No children: add current_node as a leaf node
                            print(f"    ➡️ No children found, adding {node_to_check.get('name')} as a leaf node")
                            leaf_result = create_result_node(node_to_check, root_bbox, scale, stroke_weight)
                            leaf_result['children'] = []
                            current_result['children'].append(leaf_result)
                            break
                        elif len(visible_descendants) > 1:
                            # Multiple children: add them to result, stop traversal
                            print(f"    ➡️ Multiple children found, adding them as children of {node_to_check.get('name')}")
                            for desc in visible_descendants:
                                rel_bbox = calculate_relative_bounding_box(desc, root_bbox, scale, stroke_weight)
                                desc_result = create_result_node(desc, root_bbox, scale, stroke_weight)
                                desc_result['children'] = []
                                current_result['children'].append(desc_result)
                            break
                        else:
                            node_to_check = visible_descendants[0]

            else:
                print(f"  ⬇️ CONTINUING DEEPER: Traversing into {len(children_data)} children and their descendants")
                for child, rel_bbox in children_data:
                    # Create result node for this child
                    child_result = create_result_node(child, root_bbox, scale, stroke_weight)
                    current_result['children'].append(child_result)
                    
                    # Add to queue for further processing (to find their descendants)
                    child_frame_bbox = child.get('absoluteBoundingBox')
                    if child_frame_bbox:
                        queue.append((child, child_result, child_frame_bbox))
        
    return result_root

def process_figma_screen(figma_json: Dict[str, Any], scale: float = 1.0, stroke_weight: float = 0.0) -> Dict[str, Any]:
    """
    Main function to process Figma screen JSON
    
    Args:
        figma_json: The Figma JSON data (should be the "Out of Box" frame or similar)
        scale: Scale factor for bounding box calculations
        stroke_weight: Stroke weight adjustment
    
    Returns:
        Dict: Processed JSON structure
    """
    
    print("="*60)
    print("🚀 Starting Figma JSON Hybrid Traversal")
    print("="*60)
    
    try:
        print(f"Process : Stroke weight: {stroke_weight}, Scale: {scale}")
        result = hybrid_figma_traversal(figma_json, scale, stroke_weight)
        
        print("="*60)
        print("✅ Traversal Complete!")
        print(f"Result structure built for: {result.get('name')} ({result.get('type')})")
        print("="*60)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during traversal: {str(e)}")
        raise


# def crop_image(image_path, x, y, width, height, output_path):
#     """
#     Simple function to crop image with bounding box (x, y, width, height)
    
#     Args:
#         image_path (str): Path to input image
#         x, y (int): Top-left corner coordinates
#         width, height (int): Width and height of crop area
#         output_path (str): Path to save cropped image
    
#     Example:
#         crop_image("input.png", 100, 100, 200, 150, "cropped.png")
#     """
    
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     # Calculate x2, y2 from width and height
#     # x1 = int(x)
#     # y1 = int(y)
#     # x2 = x1 + int(width)
#     # y2 = y1 + int(height)

#     x1 = x
#     y1 = y
#     x2 = x1 + width
#     y2 = y1 + height
    
#     # Ensure coordinates are within image bounds
#     img_height, img_width = img.shape[:2]
#     x1 = max(0, min(x1, img_width))
#     y1 = max(0, min(y1, img_height))
#     x2 = max(0, min(x2, img_width))
#     y2 = max(0, min(y2, img_height))
    
#     # Convert to integers for array slicing
#     x1 = int(x1)
#     y1 = int(y1)
#     x2 = int(x2)
#     y2 = int(y2)
#     # Ensure valid crop area
#     if x2 <= x1 or y2 <= y1:
#         raise ValueError(f"Invalid bounding box: x={x}, y={y}, width={width}, height={height}")
    
#     # Crop image using array slicing
#     cropped = img[y1:y2, x1:x2]
    
#     # Save cropped image
#     cv2.imwrite(output_path, cropped)
#     print(f"Cropped image saved to: {output_path}")
    
#     return output_path

def crop_image(image_path, x, y, width, height, output_path):
    """
    Simple function to crop image with bounding box (x, y, width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    x1 = x
    y1 = y
    x2 = x1 + width
    y2 = y1 + height

    img_height, img_width = img.shape[:2]
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bounding box: x={x}, y={y}, width={width}, height={height}")
        return '',False

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    patch_width = int(x2 - x1)
    patch_height = int(y2 - y1)

    # Use getRectSubPix for sub-pixel accurate cropping
    cropped = cv2.getRectSubPix(img, (patch_width, patch_height), (center_x, center_y))

    # Check if cropped image is valid
    if cropped is None or cropped.size == 0:
        print(f"❌ Cropped image is empty for bbox x={x}, y={y}, w={width}, h={height}")
        return '',False

    cv2.imwrite(output_path, cropped)
    print(f"Cropped image saved to: {output_path}")

    return output_path, True

def draw_bounding_boxes(image_path, bounding_boxes, output_path, color=(0, 255, 0), thickness=1):
    """
    Draws bounding boxes on the image and saves the result.

    Args:
        image_path (str): Path to the input image.
        bounding_boxes (list): List of dicts with keys 'x', 'y', 'width', 'height'.
        output_path (str): Path to save the output image.
        color (tuple): Bounding box color in BGR (default: green).
        thickness (int): Line thickness (default: 2).
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    for bbox in bounding_boxes:
        bbox = bbox['relativeBoundingBox']
        x0 = int(bbox['x'])
        y0 = int(bbox['y'])
        x1 = int(bbox['x'] + bbox['width'])
        y1 = int(bbox['y'] + bbox['height'])
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        # Optionally, draw the id or name above the box if available
        if 'name' in bbox:
            cv2.putText(img, bbox['name'], (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, img)
    print(f"Saved image with bounding boxes to {output_path}")


def get_bounding_boxes_relative_to_frame(node, frame_bbox, scale=1.0, stroke_weight=0.0):
    """
    Returns their bounding boxes relative to the given frame.
    """
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

# def get_leaf_bounding_boxes_relative_to_frame(node, frame_bbox):
#     """
#     Recursively finds all leaf nodes and returns their bounding boxes relative to the given frame.
#     """

#     svg_types = {
#         "VECTOR",
#         "LINE",
#         "ELLIPSE",
#         "POLYGON",
#         "STAR",
#         "RECTANGLE",
#         "BOOLEAN_OPERATION"
#     }

#     results = []
#     children = node.get('children', [])
#     node_type = node.get('type')
#     # print(f"\n\n @@@@@@@@ NODE: {node}\n\n")
#     # print(not children)
#     if not children:
#         bbox = node.get('absoluteBoundingBox')
#         visible = node.get('visible', True)

#         if bbox and visible and node_type in svg_types:
#             rel_bbox = {
#                 'id': node.get('id'),
#                 'name': node.get('name'),
#                 'type': node_type,
#                 'relativeBoundingBox': {
#                     'x': bbox['x'] - frame_bbox['x'],
#                     'y': bbox['y'] - frame_bbox['y'],
#                     'width': bbox['width'],
#                     'height': bbox['height']
#                 }
#             }
#             results.append(rel_bbox)
#     elif node.get('visible', True):
#         for child in children:
#             results.extend(get_leaf_bounding_boxes_relative_to_frame(child, frame_bbox))
#     return results



def generate_layout_svg(bounding_boxes, frame_width, frame_height, output_path):
    """
    Generate SVG with bounding boxes for layout visualization
    """
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
    <svg width="{frame_width}" height="{frame_height}" xmlns="http://www.w3.org/2000/svg">
    '''
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    for i, box in enumerate(bounding_boxes):
        bbox = box['relativeBoundingBox']
        color = colors[i % len(colors)]
        
        # Add rectangle
        svg_content += f'''  <rect x="{bbox['x']}" y="{bbox['y']}" width="{bbox['width']}" height="{bbox['height']}" 
                fill="{color}" fill-opacity="0.3" stroke="{color}" stroke-width="2"/>
                '''
        
        # Add label
        label_x = bbox['x'] + 5
        label_y = bbox['y'] + 20
        svg_content += f'''  <text x="{label_x}" y="{label_y}" font-family="Arial" font-size="12" fill="black">
            {box.get('name', 'Unknown')} ({box.get('type', '')})
            </text>
        '''
    
    svg_content += '</svg>'
    
    with open(output_path, 'w') as f:
        f.write(svg_content)
    
    print(f"Layout SVG saved to {output_path}")
    return output_path


def parse_llm_response(llm_response):
    """
    Parse structured LLM response in JSON format
    """
    import json
    
    try:
        # Try to extract JSON from response (in case there's extra text)
        response_text = llm_response.strip()
        
        # Look for JSON block in the response
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            response_data = json.loads(json_str)
            
            decision = response_data.get('decision', '').upper()
            explanation = response_data.get('explanation', '')
            
            print(f"    📋 LLM Decision: {decision}")
            print(f"    💬 Explanation: {explanation}")
            
            if decision == 'SAME':
                return True  # Child is same as parent, need to go deeper
            elif decision == 'DIFFERENT':
                return False  # Child is different, this is the final layout
            else:
                print(f"    ⚠️ Invalid decision '{decision}', defaulting to SAME (go deeper)")
                return True
                
        else:
            print("    ❌ No JSON found in response, defaulting to SAME (go deeper)")
            return True
            
    except json.JSONDecodeError as e:
        print(f"    ❌ JSON parsing failed: {e}")
        print(f"    Raw response: {llm_response}")
        print("    Defaulting to SAME (go deeper)")
        return True
    except Exception as e:
        print(f"    ❌ Unexpected error parsing response: {e}")
        print("    Defaulting to SAME (go deeper)")
        return True


def analyze_layout_structure(current_node, all_bb, depth=0, max_depth=5):
    """
    Step-1: Analyze high level layout structure (recursive)
    """
    indent = "  " * depth
    print(f"{indent}=== ANALYZING LAYOUT (Depth {depth}) ===")
    
    # Prevent infinite recursion
    if depth >= max_depth:
        print(f"{indent}⚠️ Maximum depth reached, stopping recursion")
        return {'layout_type': 'max_depth_reached', 'components': []}
    
    # Get direct children of the current node
    direct_children = current_node.get('children', [])
    visible_children = [child for child in direct_children if child.get('visible', True)]
    
    print(f"{indent}Node: {current_node.get('name')} ({current_node.get('type')})")
    print(f"{indent}Total direct children: {len(direct_children)}")
    print(f"{indent}Visible direct children: {len(visible_children)}")
    
    current_bbox = current_node['absoluteBoundingBox']
    
    if len(visible_children) > 1:
        print(f"{indent}📋 MULTIPLE COMPONENTS DETECTED - This is our layout level!")
        
        # Get bounding boxes for direct children only
        layout_components = []
        for child in visible_children:
            if child.get('absoluteBoundingBox'):
                bbox = child['absoluteBoundingBox']
                layout_components.append({
                    'id': child.get('id'),
                    'name': child.get('name'),
                    'type': child.get('type'),
                    'relativeBoundingBox': {
                        'x': bbox['x'] - current_bbox['x'],
                        'y': bbox['y'] - current_bbox['y'],
                        'width': bbox['width'],
                        'height': bbox['height']
                    }
                })
        
        print(f"{indent}Layout components found: {len(layout_components)}")
        for comp in layout_components:
            print(f"{indent}  - {comp['name']} ({comp['type']}) at ({comp['relativeBoundingBox']['x']}, {comp['relativeBoundingBox']['y']})")
        
        # Generate SVG showing layout structure
        svg_path = f"{current_node['name'].replace(' ', '_')}_layout_depth_{depth}.svg"
        generate_layout_svg(layout_components, current_bbox['width'], current_bbox['height'], svg_path)
        
        return {
            'layout_type': 'multiple_components',
            'components': layout_components,
            'svg_path': svg_path,
            'depth': depth,
            'final_node': current_node
        }
    
    elif len(visible_children) == 1:
        print(f"{indent}🔍 SINGLE COMPONENT DETECTED - Analyzing with LLM")
        
        single_child = visible_children[0]
        child_bbox = single_child.get('absoluteBoundingBox')
        
        if child_bbox:
            # Check if child covers most of the current node
            current_area = current_bbox['width'] * current_bbox['height']
            child_area = child_bbox['width'] * child_bbox['height']
            coverage_ratio = child_area / current_area if current_area > 0 else 0
            
            print(f"{indent}Child component: {single_child.get('name')} ({single_child.get('type')})")
            print(f"{indent}Coverage ratio: {coverage_ratio:.2%}")
            
            # Prepare for LLM analysis
            prompt = f"""
            You are analyzing a UI component structure to determine layout hierarchy.
            
            COMPONENT DETAILS:
            - Parent: {current_node.get('name')} ({current_node.get('type')})
            - Parent Size: {current_bbox['width']}x{current_bbox['height']}
            - Single Child: {single_child.get('name')} ({single_child.get('type')})
            - Child Size: {child_bbox['width']}x{child_bbox['height']}
            - Coverage: {coverage_ratio:.2%} of parent area
            
            TASK: Determine if this child component is the SAME as its parent or DIFFERENT.
            
            DEFINITIONS:
            - SAME: The child is just a wrapper/container with similar purpose as parent. We should analyze the child's children to find actual layout components.
            - DIFFERENT: The child represents distinct, meaningful layout content. This is our final layout level.
            
            RESPONSE FORMAT (JSON only):
            {{
                "decision": "SAME" or "DIFFERENT",
                "explanation": "Brief explanation of your reasoning"
            }}
            
            Respond with ONLY the JSON object, no additional text.
            """
            
            # Call LLM for analysis
            try:
                print(f"{indent}🤖 Calling LLM for analysis...")
                llm_response = gemini_llm_call(prompt, "appmod_s1.png")
                print(f"{indent}📝 Raw LLM Response: {llm_response}")
                
                # Parse structured LLM response
                is_same = parse_llm_response(llm_response)
                
                if is_same:
                    print(f"{indent}🔄 Decision: SAME - Going deeper into child's children")
                    # Recursively analyze the child's children
                    return analyze_layout_structure(single_child, all_bb, depth + 1, max_depth)
                else:
                    print(f"{indent}✅ Decision: DIFFERENT - This is the final layout level")
                    # This single child is the final layout
                    layout_component = {
                        'id': single_child.get('id'),
                        'name': single_child.get('name'),
                        'type': single_child.get('type'),
                        'relativeBoundingBox': {
                            'x': child_bbox['x'] - current_bbox['x'],
                            'y': child_bbox['y'] - current_bbox['y'],
                            'width': child_bbox['width'],
                            'height': child_bbox['height']
                        }
                    }
                    
                    return {
                        'layout_type': 'single_final_component',
                        'components': [layout_component],
                        'depth': depth,
                        'llm_response': llm_response,
                        'final_node': current_node
                    }
                    
            except Exception as e:
                print(f"{indent}❌ LLM analysis failed: {e}")
                print(f"{indent}⚠️ Defaulting to SAME - going deeper")
                # Default to going deeper if LLM fails
                return analyze_layout_structure(single_child, all_bb, depth + 1, max_depth)
    
    else:
        print(f"{indent}⚠️ NO VISIBLE COMPONENTS DETECTED")
        return {
            'layout_type': 'empty',
            'components': [],
            'depth': depth
        }




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

# def process_node(node, complete_tree, level=0):
#     node_id = node.get('id', '')
#     node_name = node.get('name', 'Unnamed')
#     node_type = node.get('type', 'Unknown')
#     try:
#         # Pass complete_tree to get_node_fill_color for parent lookup
#         colour = get_node_fill_color(node, complete_tree)
#         print(f"Processing node: {node_name} ({node_id}) - Type: {node_type}, Color: {colour}")
#         export_figma_nodes(
#             file_key=file_key,
#             node_ids=[node_id],
#             access_token=access_token,
#             format="png",
#             scale=1.0,
#             output_dir=OUTPUT_DIR
#         )
#         image_path = f"{OUTPUT_DIR}/{node_id}.png"
#         if os.path.exists(image_path):
#             fix_transparent_background(image_path, colour)
#         print(f"Downloaded: {node_name} ({node_id})")
#     except Exception as e:
#         print(f"Error downloading {node_id}: {e}")
    
#     # Process children
#     for child in node.get('children', []):
#         process_node(child, complete_tree, level + 1)


def collect_nodes_for_export(node):
    """Recursively collect all nodes in the tree for export."""
    nodes = [node]
    for child in node.get('children', []):
        nodes.extend(collect_nodes_for_export(child))
    return nodes

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
        image_path = f"{output_dir}/{node_id}.png"
        if os.path.exists(image_path):
            fix_transparent_background(image_path, colour)
        print(f"Downloaded: {node_name} ({node_id})")
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


if __name__ == "__main__":

    start_time = time.time()

    # Here we need to call figma api
    # figma_url = 'https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=288-8888&t=TilZbAQgV0BDqKNi-4'
    # figma_url = 'https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=204-6562&t=dMnGxshxxozW66yo-4'


    figma_url = "https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=667-17232&t=zNfxGel27IgVTEwo-4"

    file_key = extract_file_key(figma_url)
    print(file_key)

    if file_key:
        url     = f'https://api.figma.com/v1/files/{file_key}'
        payload = {}

        access_token = 'figu_fEgEwJ5kyk8h_R0_rNezor4j4BH80difoFXauIFk'
        is_api = True

        if is_api:
            headers = headers = {
            "Authorization": f"Bearer {access_token}"
                }
        else:
            headers = {'X-FIGMA-TOKEN': access_token}


        res = requests.get(url, data=payload, headers=headers)
        # print(type(res.json()))


        # Simulate the node structure as in your example
        with open('screen_data_crazy.json', 'w') as f:
            json.dump(res.json(), f)

        res_json = res.json()


        # ---------------------

        with open('screen_data_crazy.json', 'r') as f:
            res_json = json.loads(f.read())


        print(f"\nTIME TAKEN TO GET DATA FROM FIGMA: {time.time() - start_time} seconds\n\n")


        start_time = time.time()

        # Call the function
        input_page_name = 'Baptist '
        page_nodes = res_json.get("document").get('children')

        page_node = [i for i in page_nodes if i.get('name') == input_page_name][0]

        for section in page_node.get('children'):
            # print(f"Section name: {section.get('name')}\n\n")
            # if section.get("name") == "Section 2608548":
            if section.get("name") == "Section Crazy Figma":
                print("Entered\n\n")
                section_frame_node = section.get('children')
                for i in section_frame_node:
                    if i.get('name') == 'appmod_s1':
                        frame_node = i
                        print(f"Found frame node: {frame_node.get('name')} ({frame_node.get('id')})")
                        break

                with open(f"{section['name'].replace(' ', '_')}_json.json", "w") as f:
                    json.dump(frame_node, f)



                # Load the complete frame structure (for parent color lookup)
                with open(f"{section['name'].replace(' ', '_')}_json.json", "r") as f:
                    frame_json = json.load(f)
                
                




                # stroke_weight = frame_node.get('strokeWeight', 0.0)

                # export_settings = frame_node.get('exportSettings', [])
                # if export_settings and 'constraint' in export_settings[0]:
                #     constraint = export_settings[0]['constraint']
                #     if constraint.get('type') == 'SCALE':
                #         scale = constraint.get('value', 1.0)
                #     else:
                #         scale = 1.0
                # else:
                #     scale = 1.0

                # print(f"Frame Node Stroke Weight: {stroke_weight}")
                # print(f"Frame Node Scale: {scale}")

                # result = process_figma_screen(
                #     frame_node,
                #     scale=scale,
                #     stroke_weight=stroke_weight
                # )
                # with open(f"appmod_s1_result.json", "w") as f:
                #     json.dump(result, f, indent=2)
                
                # print(f"Processed result saved to appmod_s1_result.json")

                # Load the specific nodes to download
                with open(f"appmod_s1_enhanced_parallel_result.json", "r") as f:
                    result_nodes = json.load(f)
                
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                # process_node(result_nodes, frame_json, level=0)
                
                # Process the result nodes but use frame_json for parent color lookup
                process_node(result_nodes, frame_json, file_key, access_token)
            
                
                # Step-1: Find High Level Layout
                ## First check if there is only one children or mutliple. 
                ### If multiple --> treat them as components in the layout --> find their bounding boxes and generate an svg with the bounding boxes
                ### If only one --> send the bounding box to llm and just check if its inner container or it is same the image. This will be done my llm (gemini_llm_call(prompt, image_path))


                # all_bb = get_bounding_boxes_relative_to_frame(frame_node, frame_node['absoluteBoundingBox'],scale=scale, stroke_weight=stroke_weight)


                # with open("doc_sample.json", "w") as f:
                #     json.dump(all_bb, f)

                # draw_bounding_boxes("appmod_s1.png", all_bb, "appmod_s1_output_with_all_boxes.png")


                print(f"\nTIME TAKEN TO GET PROPER HEIRARCHY: {time.time() - start_time} seconds\n\n")



                # # === STEP-1: FIND HIGH LEVEL LAYOUT ===
                # print("🚀 Starting Step-1: Find High Level Layout")
                # layout_analysis = analyze_layout_structure(frame_node, all_bb, depth=0)
                
                # # Save analysis results
                # with open(f"{section['name'].replace(' ', '_')}_layout_analysis.json", "w") as f:
                #     json.dump(layout_analysis, f, indent=2)
                
                # print(f"\n✅ Layout Analysis Complete!")
                # print(f"   Layout Type: {layout_analysis['layout_type']}")
                # print(f"   Components Found: {len(layout_analysis.get('components', []))}")
                # print(f"   Analysis Depth: {layout_analysis.get('depth', 0)}")
                
                # if layout_analysis.get('svg_path'):
                #     print(f"   Layout SVG: {layout_analysis['svg_path']}")
                
                # print("="*50)