# Explainable Multi-Source AI Framework for DBS Candidacy Screening in Parkinson's Disease

> Kartic | Gachon University

## Interactive Demo

Try the live screening tool: [https://huggingface.co/spaces/kartic03/dbs-candidacy-screening](https://huggingface.co/spaces/kartic03/dbs-candidacy-screening)

## Key Results

| Dataset | n | Modality | Model | AUC-ROC | Evaluation |
|---------|---|----------|-------|---------|------------|
| WearGait-PD | 82 | Clinical (real DBS labels) | 7-feat pre-registered SVM | 0.903 | LOOCV |
| PADS | 355 | Wearable IMU (100 Hz) | XGBoost | 0.859 | 5-fold CV |
| GaitPDB | 165 | Gait force plates (100 Hz) | XGBoost | 0.996 | 5-fold CV |
| UCI Voice | 195 | Acoustic voice features | MLP | 0.953 | 5-fold CV |

The primary model uses 7 clinical features pre-registered from DBS surgical guidelines (CAPSIT-PD, Medicare criteria). It outperforms all published DBS screening tools: FLASQ-PD (AUC 0.629), STIMULUS (AUC 0.809), and DBS-PREDICT (AUC 0.79).

## Project Structure

```
.
├── preprocessing/       # Data preprocessing scripts
├── models/              # Model architectures
├── training/            # Training and evaluation scripts
├── xai/                 # Explainability (SHAP, LIME, Groq LLM)
├── analysis/            # Statistical tests and visualization
├── webapp/              # Gradio web application
│   ├── gradio_app_v2.py
│   └── svm_7feat_*.joblib
├── results/
│   ├── figures/         # Manuscript figures
│   └── tables/          # Result tables
├── requirements.txt
└── config.yaml.example
```

## Datasets

All datasets are publicly available:

| Dataset | Source | URL |
|---------|--------|-----|
| WearGait-PD | FDA CDRH | [Link](https://cdrh-rst.fda.gov/weargait-pd-wearables-dataset-gait-parkinsons-disease-and-age-matched-controls) |
| PADS | PhysioNet | [Link](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) |
| GaitPDB | PhysioNet | [Link](https://physionet.org/content/gaitpdb/1.0.0/) |
| UCI Parkinson's | UCI MLR | [Link](https://archive.ics.uci.edu/dataset/174/parkinsons) |

## Setup

```bash
conda create -n jbi_dbs python=3.11
conda activate jbi_dbs
pip install -r requirements.txt
cp config.yaml.example config.yaml  # Add your Groq API key
```

## Running the Web App

```bash
cd webapp
python gradio_app_v2.py
# Opens at http://localhost:7860
```

## License

This project is for research purposes only. Not intended for clinical decision-making without proper validation.
