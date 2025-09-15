import cv2
import json
from typing import List, Dict, Tuple, Optional

def extract_bounding_boxes(figma_data: Dict, include_names: bool = True) -> List[Dict]:
    """
    Recursively extracts all bounding boxes from Figma JSON data.
    
    Args:
        figma_data (dict): The Figma JSON data structure
        include_names (bool): Whether to include element names in the output
    
    Returns:
        list: List of dictionaries containing bounding box info and metadata
    """
    bounding_boxes = []
    
    def extract_recursive(element: Dict, depth: int = 0):
        # Extract bounding box if it exists
        if 'relativeBoundingBox' in element:
            bbox_info = {
                'relativeBoundingBox': element['relativeBoundingBox'],
                'id': element.get('id', ''),
                'name': element.get('name', ''),
                'type': element.get('type', ''),
                'depth': depth
            }
            bounding_boxes.append(bbox_info)
        
        # Recursively process children
        if 'children' in element:
            for child in element['children']:
                extract_recursive(child, depth + 1)
    
    extract_recursive(figma_data)
    return bounding_boxes

def draw_figma_bounding_boxes(image_path: str, figma_json_data: Dict, output_path: str, 
                             color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2,
                             show_labels: bool = True, label_color: Tuple[int, int, int] = (0, 255, 0),
                             min_size: int = 10, max_depth: Optional[int] = None) -> None:
    """
    Draws bounding boxes from Figma JSON data on an image and saves the result.

    Args:
        image_path (str): Path to the input image.
        figma_json_data (dict): Figma JSON data containing bounding box information.
        output_path (str): Path to save the output image.
        color (tuple): Bounding box color in BGR (default: green).
        thickness (int): Line thickness (default: 2).
        show_labels (bool): Whether to show element names as labels (default: True).
        label_color (tuple): Text color for labels in BGR (default: green).
        min_size (int): Minimum width/height to draw a bounding box (default: 10).
        max_depth (int): Maximum depth level to draw (None for all levels).
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Extract all bounding boxes from the Figma data
    bounding_boxes = extract_bounding_boxes(figma_json_data)
    
    print(f"Found {len(bounding_boxes)} elements with bounding boxes")
    
    drawn_count = 0
    for bbox_info in bounding_boxes:
        bbox = bbox_info['relativeBoundingBox']
        print(f"Trying to draw: {bbox_info}")  # Add this line
        
        # Skip if depth exceeds maximum
        if max_depth is not None and bbox_info['depth'] > max_depth:
            continue
            
        # Skip very small elements
        if bbox['width'] < min_size or bbox['height'] < min_size:
            continue
        
        # Calculate rectangle coordinates
        x0 = int(bbox['x'])
        y0 = int(bbox['y'])
        x1 = int(bbox['x'] + bbox['width'])
        y1 = int(bbox['y'] + bbox['height'])
        
        # Check if coordinates are within image bounds
        img_height, img_width = img.shape[:2]
        if x1 < 0 or y1 < 0 or x0 > img_width or y0 > img_height:
            continue
        
        # Draw rectangle
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
        drawn_count += 1
        
        # # Add label if requested
        # if show_labels and bbox_info['name']:
        #     # Position label above the box, or inside if not enough space
        #     label_y = max(y0 - 5, 15)
        #     label_text = f"{bbox_info['name']} ({bbox_info['type']})"
            
        #     # Use smaller font for smaller boxes
        #     font_scale = 0.4 if min(bbox['width'], bbox['height']) < 100 else 0.6
            
        #     cv2.putText(img, label_text, (x0, label_y), 
        #                cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 1)

    cv2.imwrite(output_path, img)
    print(f"Drew {drawn_count} bounding boxes and saved image to {output_path}")

def draw_figma_bounding_boxes_from_file(image_path: str, json_file_path: str, output_path: str,
                                       **kwargs) -> None:
    """
    Convenience function to load Figma JSON from file and draw bounding boxes.
    
    Args:
        image_path (str): Path to the input image.
        json_file_path (str): Path to the Figma JSON file.
        output_path (str): Path to save the output image.
        **kwargs: Additional arguments passed to draw_figma_bounding_boxes.
    """
    with open(json_file_path, 'r') as f:
        figma_data = json.load(f)
    
    draw_figma_bounding_boxes(image_path, figma_data, output_path, **kwargs)

# Example usage
if __name__ == "__main__":
    # Example Figma JSON data (replace with your actual data)
    
    # Draw bounding boxes with different options
    try:
        # # Basic usage
        # draw_figma_bounding_boxes(
        #     image_path="screenshot.png",
        #     figma_json_data=figma_json,
        #     output_path="output_with_boxes.png"
        # )
        
        # # With custom styling and filters
        # draw_figma_bounding_boxes(
        #     image_path="screenshot.png",
        #     figma_json_data=figma_json,
        #     output_path="output_styled.png",
        #     color=(255, 0, 0),  # Red boxes
        #     thickness=3,
        #     show_labels=True,
        #     label_color=(255, 255, 0),  # Yellow labels
        #     min_size=20,  # Only show boxes >= 20px
        #     max_depth=3   # Only show first 3 levels
        # )
        
        # From JSON file
        draw_figma_bounding_boxes_from_file(
            image_path="appmod_s1.png",
            json_file_path="appmod_s1_result.json",
            output_path="appmod_s1_final_bounding_after_llm.png"
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")