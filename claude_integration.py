import json
import base64
import time
from anthropic import AnthropicVertex


def claude_image(system_prompt, prompt, chat_history=[], image_data_list=None, model='claude-3-7-sonnet@20250219', max_tokens=8000, thinking=False, thinking_budget=1100, thinkging_msgs_stuff=[]):
    """
    Send prompt and images to Claude.
    
    Args:
        system_prompt: System prompt for Claude
        prompt: Main user prompt
        chat_history: Previous conversation history
        image_data_list: List of tuples, each containing (image_path, image_description)
                         Example: [("path/to/image1.jpg", "Image 1: This is Whole Image", "image_type like image/png or image/jpeg or image/webp or None"), 
                                  ("path/to/image2.png", "Image 2: This is component image of Image1", "image_type like image/png or image/jpeg or image/webp")]
    """

    if thinkging_msgs_stuff:
        message1 = thinkging_msgs_stuff[0]
        
     
    client = AnthropicVertex(region='us-east5', project_id="proposal-auto-ai-internal")

    # client = AnthropicVertex(region='us-east5', project_id="alan-suite")

    params = {
        "max_tokens": max_tokens,
        "temperature": 1,
        "model": model,
    }

    if system_prompt:
        params["system"] = system_prompt

    content = []
    messages = []
    messages.extend(chat_history)

    # Process images if provided
    if image_data_list:
        for image_path, image_description, image_type in image_data_list:
            
            # Then add the image
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
        
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            if '.png' in image_path.lower():
                # image_media_type = "image/webp"
                if image_type:
                    image_media_type = image_type
                else:
                    image_media_type = "image/png"

            elif '.jpg' in image_path.lower() or '.jpeg' in image_path.lower():
                image_media_type = "image/jpeg"
            else:
                # Default to jpeg if extension not recognized
                image_media_type = "image/jpeg"

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_media_type,
                    "data": image_base64,
                },
            })

            # Add the image description
            if image_description:
                content.append({
                    "type": "text", 
                    "text": image_description
                })

    # print(f"\nSending prompt: {prompt}\n")
    
    # Add the main prompt at the end
    content.append({
        "type": "text", 
        "text": prompt
    })

    messages.append({
        "role": "user",
        "content": content,
    })

    params["messages"] = messages
    if thinking:
        # min thinking budget check
        if thinking_budget < 1024:
            params["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        else:
            params["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}


    attempt = 0
    while attempt < 6:
        try:

            message = client.messages.create(**params)

            # print("################## CLAUDE MESSAGES")
            # print(message)
            # print("################## CLAUDE MESSAGES")

            response = message.model_dump_json(indent=2)
            response_dictionary = json.loads(response)

            if thinking:
                # here content[0] is ThinkingBlock
                response_text = response_dictionary['content'][1]['text']

            else:
                response_text = response_dictionary['content'][0]['text']
            
            messages.append({
                "role": "assistant",
                "content": response_text,
            })
            
            return messages, response_text
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            sleepTime = 10
            print(f"Sleeping for {sleepTime} seconds")
            time.sleep(sleepTime)

    return messages, ''