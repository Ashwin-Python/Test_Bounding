import json
import os
from PIL import Image
from export_figma import export_figma_nodes, extract_file_key

# Configuration
FIGMA_URL = "https://www.figma.com/design/L2jxRXzpdJkzJ13IivxBiS/MVUTUT---Testing?node-id=667-17232&t=amuD1LWiwIIwhXXG-4"
ACCESS_TOKEN = "figu_fEgEwJ5kyk8h_R0_rNezor4j4BH80difoFXauIFk"
FILE_KEY = extract_file_key(figma_url=FIGMA_URL)
OUTPUT_DIR = "./figma_exports_appmod_s1_chrome_enhanced_parallel_new"


def process_node(node, level=0):
   """Process each node: download image and generate HTML"""
   node_id = node.get('id', '')
   node_name = node.get('name', 'Unnamed')
   node_type = node.get('type', 'Unknown')
   
   # Sanitize node_id for file path
   sanitized_node_id = node_id
   for char in '<>:"/\\|?*;':
       sanitized_node_id = sanitized_node_id.replace(char, '_')
   sanitized_node_id = sanitized_node_id.strip('. ') or "unnamed"
   
   # Download PNG for this node
   try:
       image_path = f"{OUTPUT_DIR}/{sanitized_node_id}.png"
       # colour = get_node_fill_color(node)
       # print(f"Processing node: {node_name} ({node_id}) - Type: {node_type}, Color: {colour}")
       # export_figma_nodes(
       #     file_key=FILE_KEY,
       #     node_ids=[node_id],
       #     access_token=ACCESS_TOKEN,
       #     format="png",
       #     scale=1.0,
       #     output_dir=OUTPUT_DIR
       # )
       
       # image_path = f"{OUTPUT_DIR}/{sanitized_node_id}.png"
       # if os.path.exists(image_path):
       #     fix_transparent_background(image_path, colour)
       # print(f"Downloaded: {node_name} ({node_id})")
   except Exception as e:
       print(f"Error downloading {node_id}: {e}")
       image_path = ""
   
   # Generate HTML for this node in proper tree structure
   children = node.get('children', [])
   has_children = len(children) > 0
   
   html = '<div class="tree-node">\n'
   html += '  <div class="node-content">\n'
   html += f'    <div class="node-info">\n'
   html += f'      <strong>{node_name}</strong><br>\n'
   html += f'      <small>ID: {node_id}</small><br>\n'
   html += f'      <small>Sanitized Id: {sanitized_node_id}</small><br>\n'
   html += f'      <small>{node_type}</small>\n'
   html += f'    </div>\n'
   
   if image_path and os.path.exists(image_path):
       html += f'    <img src="{image_path}" class="node-image" alt="{node_name}">\n'
   else:
       html += f'    <div class="no-image">No Image</div>\n'
   
   html += '  </div>\n'
   
   # Process children horizontally
   if has_children:
       html += '  <div class="children-container">\n'
       for child in children:
           html += process_node(child, level + 1)
       html += '  </div>\n'
   
   html += '</div>\n'
   return html

# Main execution
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load JSON
    with open('appmod_s1_chrome_enhanced_parallel_new_result.json', 'r') as f:
        data = json.load(f)
    
    # Generate HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Figma Tree Visualization</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
                overflow-x: auto;
                overflow-y: auto;
                min-height: 100vh;
            }
            
            .tree-container {
                display: flex;
                justify-content: center;
                padding: 50px;
                min-width: max-content;
                width: 100%;
                box-sizing: border-box;
            }
            
            .tree-node {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin: 10px;
                position: relative;
            }
            
            .node-content {
                background: white;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                max-width: 200px;
                text-align: center;
                position: relative;
                z-index: 2;
            }
            
            .node-info {
                margin-bottom: 10px;
            }
            
            .node-image {
                max-width: 150px;
                max-height: 100px;
                border: 1px solid #ddd;
                border-radius: 4px;
                display: block;
                margin: 5px auto 0;
            }
            
            .no-image {
                width: 80px;
                height: 40px;
                background: #f0f0f0;
                border: 1px dashed #ccc;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
                font-size: 10px;
                margin: 5px auto 0;
                border-radius: 4px;
            }
            
            .children-container {
                display: flex;
                flex-direction: row;
                justify-content: center;
                align-items: flex-start;
                margin-top: 30px;
                position: relative;
            }
            
            /* Connecting lines */
            .tree-node {
                position: relative;
            }
            
            /* Vertical line from parent to children */
            .tree-node > .children-container:before {
                content: '';
                position: absolute;
                top: -30px;
                left: 50%;
                transform: translateX(-50%);
                width: 2px;
                height: 15px;
                background: #666;
                z-index: 1;
            }
            
            .children-container {
                position: relative;
            }
            
            /* Vertical lines from horizontal line to each child */
            .children-container > .tree-node:before {
                content: '';
                position: absolute;
                top: -15px;
                left: 50%;
                transform: translateX(-50%);
                width: 2px;
                height: 15px;
                background: #666;
                z-index: 1;
            }
            
            /* For single child, extend vertical line */
            .children-container .tree-node:only-child:before {
                height: 30px;
                top: -30px;
            }
            
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 20px;
            }
            
            .scroll-wrapper {
                width: 100%;
                height: calc(100vh - 100px);
                overflow: auto;
                border: 1px solid #ddd;
                border-radius: 8px;
                background: white;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>Figma Design Tree</h1>
        <div class="tree-container">
    """
    
    # Process the root node
    html_content += process_node(data)
    
    html_content += """
        </div>
        
        <script>
        // Fix horizontal connecting lines after page loads
        window.addEventListener('load', function() {
            const containers = document.querySelectorAll('.children-container');
            
            containers.forEach(container => {
                const children = container.querySelectorAll(':scope > .tree-node');
                
                if (children.length > 1) {
                    const firstChild = children[0];
                    const lastChild = children[children.length - 1];
                    
                    const firstRect = firstChild.getBoundingClientRect();
                    const lastRect = lastChild.getBoundingClientRect();
                    const containerRect = container.getBoundingClientRect();
                    
                    const leftOffset = firstRect.left + firstRect.width/2 - containerRect.left;
                    const rightOffset = lastRect.left + lastRect.width/2 - containerRect.left;
                    
                    // Create or update horizontal line
                    const existingLine = container.querySelector('.horizontal-line');
                    if (existingLine) {
                        existingLine.remove();
                    }
                    
                    const horizontalLine = document.createElement('div');
                    horizontalLine.className = 'horizontal-line';
                    horizontalLine.style.cssText = `
                        position: absolute;
                        top: -15px;
                        left: ${leftOffset}px;
                        width: ${rightOffset - leftOffset}px;
                        height: 2px;
                        background: #666;
                        z-index: 1;
                    `;
                    
                    container.appendChild(horizontalLine);
                }
            });
        });
        </script>
    </body>
    </html>
    """
    
    # Save HTML file
    with open('figma_tree_appmod_s1_chrome_enhanced_parallel_new_result.html', 'w') as f:
        f.write(html_content)
    
    print("HTML file created: figma_tree_appmod_s1_chrome_enhanced_parallel_new_result.html")

if __name__ == "__main__":
    main()