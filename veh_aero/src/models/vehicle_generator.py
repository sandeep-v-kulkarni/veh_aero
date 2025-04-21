import boto3
import json
import base64
import os
from PIL import Image
import io
import uuid
import streamlit as st

class VehicleImageGenerator:
    def __init__(self):
        """Initialize the Bedrock client"""
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = 'amazon.nova-canvas-v1:0'
        
        self.output_dir = 'generated_vehicles'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def generate_image(self, prompt: str, negative_prompt: str = "low quality, blurry, bad anatomy"):
        """Generate vehicle image using Nova Canvas"""
        try:
            body = json.dumps({
                "taskType": "TEXT_IMAGE",
                "textToImageParams": {
                    "text": prompt,
                    "negativeText": negative_prompt
                },
                "imageGenerationConfig": {
                    "numberOfImages": 1,
                    "quality": "standard",
                    "height": 1024,
                    "width": 1024,
                    "cfgScale": 8.0
                }
            })

            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response.get("body").read())
            base64_image = response_body.get("images")[0]
            
            image_bytes = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_bytes))
            
            filename = f"vehicle_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath)
            
            return image, filepath
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None, None