# Explainable Multi-Source AI Framework for DBS Candidacy Screening in Parkinson's Disease

This repository contains the code and results for the paper:

> **"Explainable multi-source AI framework for deep brain stimulation candidacy screening in Parkinson's disease"**
>
> Kartic, Jiwon Lim, Tae-Sik Park
>
> Department of Life Sciences, Gachon University, Republic of Korea

## Interactive Demo

Try the live screening tool: [https://huggingface.co/spaces/kartic03/dbs-candidacy-screening](https://huggingface.co/spaces/kartic03/dbs-candidacy-screening)

## Key Results

| Dataset | n | Modality | Model | AUC-ROC | Evaluation |
|---------|---|----------|-------|---------|------------|
| WearGait-PD | 82 | Clinical (real DBS labels) | XGBoost Top-10 | 0.878 (0.792-0.950) | LOOCV |
| PADS | 370 | Wearable IMU (100 Hz) | XGBoost | 0.860 (0.818-0.897) | 5-fold CV |
| GaitPDB | 165 | Gait force plates (100 Hz) | XGBoost | 0.988 (0.973-0.998) | 5-fold CV |
| UCI Voice | 195 | Acoustic voice features | MLP | 0.972 (0.945-0.992) | 5-fold CV |

## Project Structure

```
.
├── preprocessing/       # Data preprocessing scripts
│   ├── pads_preprocessing.py
│   ├── voice_preprocessing.py
│   ├── gait_preprocessing.py
│   └── data_fusion.py
├── models/              # Model architectures
│   ├── wearable_encoder.py
│   ├── voice_encoder.py
│   ├── gait_encoder.py
│   ├── fusion_model.py
│   └── baseline_models.py
├── training/            # Training and evaluation scripts
│   ├── train_encoders.py
│   ├── train_clinical.py
│   ├── train_fusion.py
│   ├── evaluate.py
│   └── optimize_v2.py
├── xai/                 # Explainability (SHAP, LIME, Groq LLM)
│   ├── shap_analysis.py
│   ├── lime_analysis.py
│   └── groq_report.py
├── analysis/            # Statistical tests and visualization
│   ├── statistical_tests.py
│   └── visualization_v4.py
├── webapp/              # Gradio web application
│   ├── gradio_app_v2.py
│   └── xgb_top10_*.joblib
├── results/
│   ├── figures/         # Manuscript figures
│   └── tables/          # Result tables
├── requirements.txt
└── config.yaml.example  # Config template (add your API keys)
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

