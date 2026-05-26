## Early Detection of ASD — Text + Image Multi-Modal System

Lab project: **multi-modal ASD screening** using:

- **Image (primary)** — HOG facial features + linear classifier (`asd_images/ASD`, `asd_images/No_ASD`)
- **Text** — caregiver/clinician notes via TF-IDF + Logistic Regression (`asd_text.csv`)

Tabular questionnaire data is **not** used in this version.

### Files

| File | Purpose |
|------|---------|
| `advance_asd_detection.ipynb` | Main notebook (train + user input) |
| `advance_asd_detection_executed.ipynb` | Fully executed copy with outputs |
| `asd_text.csv` | Text notes + labels |
| `asd_images/ASD/` | ASD-labeled images |
| `asd_images/No_ASD/` | Non-ASD images |

If image folders are empty, the notebook auto-generates demo face images for training.

### Setup

```bash
pip install -r requirements.txt
jupyter notebook advance_asd_detection.ipynb
```

Run **Kernel → Restart & Run All**. Edit `example_image_path` and `example_text_note` in the user-input cell for your demo.

### Fusion

Combines image + text with strategies: `average`, `weighted` (65% image / 35% text), `voting`, `meta`.
