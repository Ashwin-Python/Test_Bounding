import base64
import os
from google import genai
from google.genai import types
from PIL import Image
import logging
import time

PROJECT_ID = "proposal-auto-ai-internal"
LOCATION = "global"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"


def gemini_image(system_prompt, prompt, chat_history=[], image_data_list=None):
    """
    Calls the Gemini LLM with images and a prompt to generate content.
    
    Args:
        system_prompt (str): The system prompt to guide the LLM.
        prompt (str): The user prompt to generate content.
        chat_history (list): List of previous chat messages (not used in this function).
        image_data_list (list): List of tuples containing image paths and descriptions. Example:
                                 [("path/to/image1.png", "Image-1"),
                                  ("path/to/image2.png", "Image-2")]
        
    Returns:
        str: The generated response text from the LLM.
    """

    # image = Image.open(image_path_1)
    # screen_width, screen_height = image.size

    print("ENTERED")

    parts_list = []
    INCLUDE_THOUGHTS = True
    response_text = ""
    max_retries = 10

    for i in range(max_retries):
        try:
            client = genai.Client(
                vertexai=True,
                project=PROJECT_ID,
                location=LOCATION,
            )

            if image_data_list:
                for image_path, image_description in image_data_list:
                    with open(image_path, "rb") as f:
                        image_data = f.read()

                    parts_list.append(
                            types.Part.from_bytes(
                            data=image_data, mime_type="image/png"
                        )
                    )

                parts_list.append(
                    types.Part.from_text(
                            text=prompt
                        )
                )

            model = "gemini-2.5-flash"

            # model = "gemini-2.5-flash-preview-04-17"
            contents = [
                types.Content(
                    role="user",
                    parts=parts_list,
                ),
            ]

            generate_content_config = types.GenerateContentConfig(
                response_mime_type="text/plain",
                thinking_config=types.ThinkingConfig(
                    include_thoughts=INCLUDE_THOUGHTS,
                ),
            )

            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                # print(chunk.text, end="")
                if chunk.text:
                    response_text += chunk.text

            return response_text

            # response_text = client.models.genera te_content(
            #     model=model,
            #     contents=contents,
            #     config=generate_content_config,
            # ).text

        except Exception as e:
            logging.info(f"Error while generating BB using LLM: {e}")
            logging.info(f"iteration-{i}, Retrying after 5 secs.....")
            time.sleep(5)

    return response_text