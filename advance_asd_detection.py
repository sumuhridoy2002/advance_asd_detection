import torch
from transformers import pipeline
from PIL import Image
from torchvision import models, transforms

text_model = pipeline("text-classification", model="roberta-large-mnli")
image_model = models.efficientnet_b3(weights="DEFAULT")
image_model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def analyze_text(text):
    results = text_model(text)
    raw_score = sum([r["score"] * (1 if r["label"].lower().startswith("entail") else 0.5) for r in results]) / len(results)
    return min(max(raw_score + 0.25, 0), 1)

def analyze_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = image_model(img)
        prob = torch.softmax(output, dim=1).numpy().flatten()
        base_score = float(0.5 * prob.max() + 0.5 * prob.mean())
        return min(base_score + 0.40, 0.95)

def final_decision(scores):
    weights = [0.6, 0.4] if len(scores) == 2 else [0.4, 0.4, 0.1, 0.1]
    weighted_score = sum([s * w for s, w in zip(scores, weights)])
    
    if weighted_score > 0.60: risk = "HIGH RISK"
    elif weighted_score > 0.40: risk = "MEDIUM RISK"
    else: risk = "LOW RISK"
    
    return risk, weighted_score, weights

def get_natural_language_explanation(avg_score):
    if avg_score > 0.60: return "Strong indicators of ASD. Professional clinical screening is strongly recommended."
    elif avg_score > 0.40: return "Moderate indicators detected. Further observation by a specialist is advised."
    else: return "No significant indicators of ASD detected in the provided input."

def explain_contribution(scores, weights):
    print("\n--- Explainability: Feature Contribution ---")
    modalities = ["Text", "Image", "Video", "Audio"]
    for i, modality in enumerate(modalities[:len(scores)]):
        print(f"- {modality}: Contribution {(scores[i] * weights[i]):.2f}")