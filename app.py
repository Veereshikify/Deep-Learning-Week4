import gradio as gr
import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession(
    "baseline_cnn.onnx",
    providers=["CPUExecutionProvider"]
)

class_names = ["cardboard", "glass", "metal", "paper", "plastic"]

recycling_tips = {
    "cardboard": "Keep cardboard clean and dry. Flatten it before recycling.",
    "glass": "Rinse glass containers and recycle them separately.",
    "metal": "Clean metal cans and place them in metal recycling.",
    "paper": "Keep paper dry and separate from contaminated materials.",
    "plastic": "Clean plastic containers and check local recycling rules."
}

def predict(image):
    if image is None:
        return "Please upload an image.", ""

    image = image.convert("RGB").resize((128, 128))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    logits = session.run(["output"], {"input": arr})[0][0]
    exp = np.exp(logits - np.max(logits))
    probabilities = exp / np.sum(exp)

    index = int(np.argmax(probabilities))
    category = class_names[index]
    confidence = float(probabilities[index]) * 100

    result = f"Predicted Category: {category.title()}\nConfidence: {confidence:.2f}%"

    if confidence < 50:
        result += "\n⚠️ Low confidence. Please try a clearer image of the waste item."

    return result, recycling_tips[category]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a waste image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Recycling Guidance")
    ],
    title="AI Waste Image Classification System",
    description="Upload an image to classify it into cardboard, glass, metal, paper, or plastic."
)

if __name__ == "__main__":
    demo.launch()
