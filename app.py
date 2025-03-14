from diffusers import StableDiffusionPipeline
import torch
from flask import Flask, render_template, request
import io
import base64
from PIL import Image

# Initialize the Stable Diffusion pipeline
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Function to generate images from a text prompt
def generate_images(prompt):
    max_length = 77  # Stable Diffusion's text encoder max token length
    if len(prompt.split()) > max_length:
        prompt = " ".join(prompt.split()[:max_length])
    results = pipe(prompt, num_inference_steps=50, guidance_scale=7.5)
    return results.images

# Function to convert images to base64 strings for HTML rendering
def convert_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Flask application setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', images=None)

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form.get("prompt")
    if not prompt:
        return render_template('index.html', error="Please enter a prompt!", images=None)
    try:
        images = generate_images(prompt)
        images_base64 = [convert_image_to_base64(image) for image in images]
        return render_template('index.html', images=images_base64)
    except Exception as e:
        return render_template('index.html', error=f"Error generating images: {str(e)}", images=None)

if __name__ == "__main__":
    app.run(debug=True)
