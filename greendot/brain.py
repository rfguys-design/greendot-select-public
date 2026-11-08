"""The ONLY place API calls happen. OWNER: Markus

The API key is read here from the environment and nowhere else. If a key
appears anywhere outside this module, that is a bug.
"""

import base64
import json
import os
import sys

import cv2
from openai import OpenAI

from greendot.config import CONFIG
from greendot.skills import RESPONSE_SCHEMA, SYSTEM_PROMPT

_client = None


def get_client():
    """Lazily create the client, failing with a clear message if the key is missing."""
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("OPENAI_API_KEY is not set. Export it and rerun. "
                     "Never hardcode it in a source file.")
        _client = OpenAI(api_key=key, timeout=CONFIG["request_timeout_s"],
                         max_retries=CONFIG["max_retries"])
    return _client


def edit_image(crop, instruction):
    """Image in, image out: the crop plus the user's own words go to the image model.

    No description step in between -- the image model looks at the drawing
    itself, so it keeps the composition and only changes what was asked.
    Returns the new image as PNG bytes.
    """
    success, encoded = cv2.imencode(".png", crop)
    if not success:
        raise ValueError("Could not encode the image as PNG.")
    response = get_client().images.edit(
        model=CONFIG["image_model"],
        image=("drawing.png", encoded.tobytes(), "image/png"),
        prompt=instruction,
        quality=CONFIG["image_quality"],
        size=CONFIG["image_size"],
        timeout=CONFIG["image_timeout_s"],   # drawing takes ~20 s, far longer than text
    )
    return base64.b64decode(response.data[0].b64_json)


def ask_model(crop, transcript):
    # Keeping all API calls here makes it easier to control API access and change models in one place.
    success, encoded = cv2.imencode(
        ".jpg",
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, CONFIG["jpeg_quality"]]
    )

    if not success:
        raise ValueError("Could not encode the image as JPEG.")

    image_data = base64.b64encode(encoded).decode("utf-8")
    intent = transcript if transcript else CONFIG["default_intent"]

    response = get_client().chat.completions.create(
        model=CONFIG["vision_model"],
        response_format=RESPONSE_SCHEMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": intent},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                            "detail": CONFIG["image_detail"]
                        }
                    }
                ]
            }
        ]
    )

    result = response.choices[0].message.content
    return json.loads(result)