import torch
import numpy as np
from transformers import pipeline
from PIL import Image
from torchvision import models, transforms
import torch.nn.functional as F

# =========================================================
# DEVICE CONFIG
# =========================================================

DEVICE = 0 if torch.cuda.is_available() else -1

# =========================================================
# LOAD MODELS
# =========================================================

text_model = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    device=DEVICE
)

image_model = models.efficientnet_b3(weights="DEFAULT")
image_model.eval()

# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# TEXT ANALYSIS
# =========================================================

def analyze_text(text):

    if not text or len(text.strip()) < 10:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "details": "Insufficient text input"
        }

    try:
        results = text_model(text)

        # Better normalization
        scores = []

        for r in results:

            label = r["label"].lower()
            score = r["score"]

            if "entail" in label:
                scores.append(score)

            elif "neutral" in label:
                scores.append(score * 0.5)

            else:
                scores.append(score * 0.2)

        final_score = float(np.mean(scores))

        confidence = float(max([r["score"] for r in results]))

        return {
            "score": round(final_score, 4),
            "confidence": round(confidence, 4),
            "details": results
        }

    except Exception as e:
        return {
            "score": 0.0,
            "confidence": 0.0,
            "details": str(e)
        }

# =========================================================
# IMAGE ANALYSIS
# =========================================================

def analyze_image(image_path):

    try:

        img = Image.open(image_path).convert("RGB")

        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():

            output = image_model(tensor)

            probabilities = F.softmax(output, dim=1)

            max_prob = probabilities.max().item()

            entropy = -torch.sum(
                probabilities * torch.log(probabilities + 1e-10)
            ).item()

            normalized_entropy = entropy / np.log(probabilities.shape[1])

            confidence = 1 - normalized_entropy

            # More stable scoring
            final_score = (max_prob * 0.7) + (confidence * 0.3)

            return {
                "score": round(float(final_score), 4),
                "confidence": round(float(confidence), 4),
                "details": {
                    "max_probability": round(max_prob, 4),
                    "entropy": round(entropy, 4)
                }
            }

    except Exception as e:

        return {
            "score": 0.0,
            "confidence": 0.0,
            "details": str(e)
        }

# =========================================================
# MULTIMODAL FUSION
# =========================================================

def final_decision(text_result, image_result):

    text_score = text_result["score"]
    image_score = image_result["score"]

    text_conf = text_result["confidence"]
    image_conf = image_result["confidence"]

    # Dynamic weighting using confidence
    total_conf = text_conf + image_conf + 1e-8

    text_weight = text_conf / total_conf
    image_weight = image_conf / total_conf

    fused_score = (
        (text_score * text_weight) +
        (image_score * image_weight)
    )

    # Risk classification
    if fused_score >= 0.75:
        risk = "HIGH RISK"

    elif fused_score >= 0.45:
        risk = "MEDIUM RISK"

    else:
        risk = "LOW RISK"

    return {
        "risk": risk,
        "score": round(float(fused_score), 4),
        "weights": {
            "text": round(float(text_weight), 4),
            "image": round(float(image_weight), 4)
        }
    }

# =========================================================
# NATURAL LANGUAGE INTERPRETATION
# =========================================================

def get_natural_language_explanation(score):

    if score >= 0.75:
        return (
            "High behavioral irregularity detected. "
            "Professional ASD screening is recommended."
        )

    elif score >= 0.45:
        return (
            "Moderate irregular patterns detected. "
            "Further clinical observation may be beneficial."
        )

    else:
        return (
            "Low irregularity detected in the provided inputs."
        )

# =========================================================
# EXPLAINABILITY
# =========================================================

def explain_contribution(final_result):

    print("\n========== FEATURE CONTRIBUTION ==========")

    weights = final_result["weights"]

    print(f"Text Contribution Weight : {weights['text']:.2f}")
    print(f"Image Contribution Weight: {weights['image']:.2f}")

    print("==========================================")

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    text_input = """
    The child demonstrates repetitive behavior,
    reduced eye contact, and communication difficulty.
    """

    image_path = "./sample.jpg"

    print("\nProcessing multimodal ASD screening...\n")

    text_result = analyze_text(text_input)

    image_result = analyze_image(image_path)

    final_result = final_decision(
        text_result,
        image_result
    )

    print("========== FINAL RESULT ==========")

    print(f"Risk Level : {final_result['risk']}")
    print(f"Confidence : {final_result['score']:.2f}")

    print("\n========== INTERPRETATION ==========")

    print(
        get_natural_language_explanation(
            final_result["score"]
        )
    )

    explain_contribution(final_result)