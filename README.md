# Early Detection of Autism Spectrum Disorder (ASD) Using Multi-Modal Machine Learning

This project presents a **multi-modal machine learning system** for the early screening of **Autism Spectrum Disorder (ASD)** using both:

- **Image data**
- **Textual behavioral notes**

The system independently trains models for each modality and combines their predictions using multiple fusion strategies to generate a final ASD / No ASD prediction.

> NOTE:
> This project is developed for academic and research purposes only.
> It is NOT intended for real medical diagnosis.

---

# Project Objectives

The main objectives of this project are:

- Build separate machine learning models for image and text datasets
- Combine predictions using multimodal fusion
- Accept user input dynamically
- Display ASD / No ASD prediction
- Provide confidence score and natural language interpretation
- Demonstrate explainable AI concepts using feature importance

---

# Modalities Used

## 1. Image Modality (Primary)

The image model uses:

- Facial image dataset
- HOG (Histogram of Oriented Gradients) feature extraction
- SVM classifier (`SVC`)

Dataset structure:

```text
asd_images/
│
├── Train/
│   ├── autism/
│   ├── autistic/
│   └── tipical/
```

Labels:
- `autism`, `autistic` → ASD
- `tipical` → No ASD

---

## 2. Text Modality

The text model uses:

- Behavioral / caregiver / clinician notes
- TF-IDF vectorization
- Logistic Regression classifier

Dataset file:

```text
asd_text.csv
```

Required columns:

| Column | Description |
|---|---|
| `note` | Behavioral text |
| `label` | ASD label (0 or 1) |

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Scikit-learn | Machine learning |
| OpenCV / PIL | Image processing |
| scikit-image | HOG feature extraction |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Joblib | Model saving |

---

# Machine Learning Models

| Modality | Feature Extraction | Model |
|---|---|---|
| Image | HOG | SVM (`SVC`) |
| Text | TF-IDF | Logistic Regression |

---

# Fusion Strategies

The system combines image and text predictions using multiple methods:

| Method | Description |
|---|---|
| `average` | Mean probability averaging |
| `weighted` | Weighted averaging (65% image + 35% text) |
| `voting` | Majority-based decision |
| `meta` | Custom fusion strategy |

Weighted fusion gives higher importance to the image modality because it achieved better validation performance during experimentation.

---

# Features

- Multi-modal ASD screening
- Image + text prediction
- User input system
- Multiple fusion strategies
- Confidence score generation
- Natural language interpretation
- Confusion matrix visualization
- Feature importance analysis
- Model comparison graph
- Saved trained models

---

# Evaluation Metrics

The models are evaluated using:

- Accuracy
- F1 Score
- Classification Report
- Confusion Matrix

---

# Explainability

The project includes a basic explainability module using:

- Text feature importance analysis

The system identifies important ASD-related words learned by the Logistic Regression classifier.

---

# Project Structure

```text
project/
│
├── advance_asd_detection.ipynb
├── requirements.txt
├── asd_text.csv
│
├── asd_images/
│   └── Train/
│       ├── autism/
│       ├── autistic/
│       └── tipical/
│
├── image_model.pkl
├── text_model.pkl
├── image_scaler.pkl
│
└── README.md
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook advance_asd_detection.ipynb
```

Then open:

```text
Kernel → Restart & Run All
```

---

# User Prediction System

The notebook accepts:

- User image path
- User behavioral note
- Fusion method

Example:

```python
user_img = "test.jpg"

user_txt = "Child avoids eye contact and repetitive behavior observed."

fusion_method = "weighted"
```

Output:

```text
Prediction: ASD
Confidence: 82%
Explanation:
Moderate ASD-related indicators were detected.
```

---

# Visualizations Included

- Class distribution graphs
- Confusion matrices
- Model comparison charts

---

# Saved Models

The trained models are automatically saved as:

```text
image_model.pkl
text_model.pkl
image_scaler.pkl
```

---

# Limitations

This system is an educational prototype and has several limitations:

- ASD cannot be accurately diagnosed solely from facial images or text notes
- Dataset quality and size may affect performance
- The system should not be used for clinical diagnosis

---

# Future Improvements

Possible future enhancements:

- CNN / Deep Learning image models
- Audio modality integration
- SHAP / LIME explainability
- Web application deployment
- Larger clinical datasets
- Real-time screening interface

---

# Conclusion

This project demonstrates how multimodal machine learning can combine image and text information for ASD screening tasks.

By integrating separate AI models and fusion techniques, the system achieves more informative predictions than single-modality approaches while also providing explainable outputs and confidence estimation.

---

# Author

**Ali Azgor Hridoy**  

GitHub: https://github.com/sumuhridoy2002

Lab Final Project — Early Detection of Autism Spectrum Disorder (ASD) Using Multi-Modal Machine Learning