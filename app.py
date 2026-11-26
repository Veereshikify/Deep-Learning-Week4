import os
import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]
DEVICE = torch.device("cpu")

model = BaselineCNN().to(DEVICE)
model.load_state_dict(torch.load("baseline_cnn.pth", map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

DESCRIPTIONS = {
    "cardboard": "Paper-based packaging material. Recycle as cardboard.",
    "glass": "Glass material. Recycle as glass.",
    "metal": "Metal material. Recycle as metal.",
    "paper": "Paper material. Recycle as paper.",
    "plastic": "Plastic material. Recycle as plastic."
}

def predict_waste(image):
    """Classify an uploaded waste image and return category, confidence, recycling guidance, and warning."""
    if image is None:
        return "No image", "0.00%", "Please upload an image.", "Waiting for an image."
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax().item())
    category = CLASS_NAMES[index]
    confidence = float(probabilities[index].item() * 100)
    warning = (
        "Low confidence — please verify the material manually."
        if confidence < 50
        else "Prediction generated. Please check the result before recycling."
    )
    return category.capitalize(), f"{confidence:.2f}%", DESCRIPTIONS[category], warning

demo = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(type="pil", label="Upload Waste Image"),
    outputs=[
        gr.Textbox(label="Predicted Waste Category"),
        gr.Textbox(label="Confidence Score"),
        gr.Textbox(label="Recycling Information"),
        gr.Textbox(label="Prediction Warning")
    ],
    title="♻️ Smart Waste Classification System",
    description="Upload an image to classify cardboard, glass, metal, paper, or plastic. Confidence is a model score, not a guarantee.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port)
