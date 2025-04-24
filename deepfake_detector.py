import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load model and processor
processor = AutoImageProcessor.from_pretrained("dima806/deepfake_vs_real_image_detection")
model = AutoModelForImageClassification.from_pretrained("dima806/deepfake_vs_real_image_detection")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def detect_deepfake(image_path):
    try:
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        # Get label and confidence
        label = model.config.id2label[predicted_class.item()].lower()
        confidence = confidence.item()

        # Convert label to boolean (True for fake, False for real)
        is_fake = label == "fake"

        return {
            "is_fake": is_fake,
            "confidence": confidence
        }
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")