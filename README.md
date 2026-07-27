# Explainable prediction of recorded deep brain stimulation status in Parkinson's disease

> Kartic | Gachon University

## Interactive Demo

Try the live demo (research prototype, not for clinical use): [https://huggingface.co/spaces/kartic03/dbs-candidacy-screening](https://huggingface.co/spaces/kartic03/dbs-candidacy-screening)

## Key Results

| Dataset | n | Modality | Model | AUC-ROC | Evaluation |
|---------|---|----------|-------|---------|------------|
| WearGait-PD (primary, recorded DBS status) | 82 | Clinical | 7-feature pre-registered SVM | 0.903 | LOOCV |
| PADS (exploratory, PD vs control) | 355 | Wearable IMU (100 Hz) | XGBoost | 0.859 | 5-fold CV |
| GaitPDB (exploratory, PD vs control) | 165 | Gait force plates (100 Hz) | XGBoost | 0.996 | 5-fold CV |
| UCI Voice (exploratory, PD vs control) | 195 | Acoustic voice features | MLP | 0.953 | 5-fold CV |

The primary model uses 7 clinical features pre-registered from DBS surgical guidelines (CAPSIT-PD, Medicare criteria) and predicts recorded DBS status (whether a patient had received DBS), which is distinct from validated DBS candidacy. Benchmarked against published figures, its LOOCV AUC (0.903) exceeds the reported values of FLASQ-PD (0.629) and STIMULUS (0.809); those come from different cohorts with different outcome definitions, so this is a comparison against published figures rather than a same-population head-to-head. The three additional datasets carry PD-vs-control labels only and are exploratory. This is preliminary work; clinical use would require external, prospective, DBS-labelled validation.

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
