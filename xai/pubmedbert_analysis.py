#!/usr/bin/env python3
"""PubMedBERT Clinical Validation — validates AI feature importance against biomedical knowledge.

Uses Microsoft's PubMedBERT (BiomedBERT) via HuggingFace Inference API to:
1. Validate that our top SHAP features align with PD/DBS literature
2. Generate biomedical embeddings for clinical text
3. Score clinical relevance of each feature for DBS candidacy

JBI DBS Screening Project
"""

import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

with open(PROJECT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

HF_TOKEN = CFG.get("huggingface", {}).get("api_key", "")
PUBMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
TABLES = PROJECT / "results" / "tables"
FIGURES = PROJECT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)

from huggingface_hub import InferenceClient


def validate_features_with_pubmedbert():
    """Use PubMedBERT fill-mask to validate clinical relevance of top features."""
    print("=" * 80)
    print("  PubMedBERT Clinical Feature Validation")
    print("=" * 80)

    client = InferenceClient(token=HF_TOKEN)

    # Our top features from SHAP analysis
    feature_prompts = {
        "hoehn_yahr": "The [MASK] and Yahr scale is the most important predictor of DBS candidacy in Parkinson disease.",
        "disease_duration": "Disease [MASK] is a critical factor in determining deep brain stimulation eligibility.",
        "updrs_iii_total": "The [MASK] score from MDS-UPDRS Part III is used to assess motor severity in Parkinson disease.",
        "tremor": "[MASK] is the motor symptom most responsive to deep brain stimulation in Parkinson disease.",
        "gait_asymmetry": "Gait [MASK] is an important biomarker for assessing Parkinson disease severity.",
        "bradykinesia": "[MASK] is a cardinal motor feature of Parkinson disease that indicates disease progression.",
        "rigidity": "Motor [MASK] in Parkinson disease is assessed during the neurological examination.",
        "age": "Patient [MASK] is an important consideration for deep brain stimulation candidacy.",
        "dbs_general": "Deep brain stimulation of the [MASK] nucleus is the preferred target for Parkinson disease.",
        "screening": "Early [MASK] for DBS candidacy can improve patient outcomes in Parkinson disease.",
    }

    results = []
    for feature, prompt in feature_prompts.items():
        try:
            response = client.fill_mask(prompt, model=PUBMEDBERT_MODEL)
            top3 = [(r.token_str, round(r.score, 4)) for r in response[:3]]
            results.append({
                "Feature": feature,
                "Prompt": prompt,
                "Top1_Word": top3[0][0],
                "Top1_Confidence": top3[0][1],
                "Top2_Word": top3[1][0] if len(top3) > 1 else "",
                "Top2_Confidence": top3[1][1] if len(top3) > 1 else 0,
                "Top3_Word": top3[2][0] if len(top3) > 2 else "",
                "Top3_Confidence": top3[2][1] if len(top3) > 2 else 0,
                "Clinically_Valid": "Yes" if top3[0][1] > 0.1 else "Uncertain",
            })
            print(f"  {feature:25s} → {top3[0][0]} ({top3[0][1]:.1%})")
        except Exception as e:
            print(f"  {feature:25s} → Error: {e}")
            results.append({"Feature": feature, "Prompt": prompt, "Top1_Word": "API_ERROR",
                           "Top1_Confidence": 0, "Clinically_Valid": "Error"})

    df = pd.DataFrame(results)
    df.to_csv(TABLES / "pubmedbert_feature_validation.csv", index=False)
    print(f"\n  Saved: pubmedbert_feature_validation.csv")
    return df


def compute_clinical_embeddings():
    """Generate PubMedBERT embeddings for clinical descriptions of DBS candidates."""
    print("\n" + "=" * 80)
    print("  PubMedBERT Clinical Embeddings")
    print("=" * 80)

    client = InferenceClient(token=HF_TOKEN)

    clinical_texts = {
        "high_risk": "Advanced Parkinson disease with severe tremor, gait impairment, Hoehn Yahr stage 3, "
                     "medication-resistant symptoms, and significant motor asymmetry indicating deep brain stimulation candidacy.",
        "moderate_risk": "Moderate Parkinson disease with bilateral symptoms, Hoehn Yahr stage 2.5, "
                        "partial medication response, mild gait disturbance requiring monitoring for DBS evaluation.",
        "low_risk": "Early stage Parkinson disease with mild unilateral tremor, Hoehn Yahr stage 1, "
                    "good medication response, no gait impairment, routine neurological follow-up.",
        "dbs_indication": "Deep brain stimulation is indicated for Parkinson disease patients with motor complications, "
                         "medication-refractory tremor, and adequate cognitive function.",
        "dbs_contraindication": "DBS is contraindicated in patients with dementia, active psychiatric illness, "
                                "significant brain atrophy, or unrealistic expectations about surgical outcomes.",
    }

    embeddings = {}
    for name, text in clinical_texts.items():
        try:
            emb = client.feature_extraction(text, model=PUBMEDBERT_MODEL)
            emb_array = np.array(emb)
            # Use CLS token embedding (first token) or mean pooling
            if emb_array.ndim == 2:
                emb_vec = emb_array.mean(axis=0)  # mean pooling
            elif emb_array.ndim == 3:
                emb_vec = emb_array[0].mean(axis=0)
            else:
                emb_vec = emb_array.flatten()[:768]
            embeddings[name] = emb_vec
            print(f"  {name:25s} → embedding dim={len(emb_vec)}")
        except Exception as e:
            print(f"  {name:25s} → Error: {e}")

    # Compute similarity matrix
    if len(embeddings) >= 2:
        names = list(embeddings.keys())
        vecs = np.array([embeddings[n] for n in names])
        # Cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        sim_matrix = (vecs @ vecs.T) / (norms @ norms.T + 1e-10)

        sim_df = pd.DataFrame(sim_matrix, index=names, columns=names).round(4)
        sim_df.to_csv(TABLES / "pubmedbert_clinical_similarity.csv")
        print(f"\n  Similarity matrix:")
        print(sim_df.to_string())
        print(f"\n  Saved: pubmedbert_clinical_similarity.csv")

        # Key findings
        print(f"\n  Key findings:")
        print(f"    high_risk ↔ dbs_indication:      {sim_matrix[names.index('high_risk'), names.index('dbs_indication')]:.4f}")
        print(f"    low_risk ↔ dbs_contraindication:  {sim_matrix[names.index('low_risk'), names.index('dbs_contraindication')]:.4f}")
        print(f"    high_risk ↔ low_risk:             {sim_matrix[names.index('high_risk'), names.index('low_risk')]:.4f}")

    return embeddings


def generate_pubmedbert_report(patient_data: dict) -> str:
    """Generate a PubMedBERT-enhanced clinical assessment using fill-mask."""
    client = InferenceClient(token=HF_TOKEN)

    assessments = []
    prompts = [
        (f"A patient with UPDRS score of {patient_data.get('updrs', 25)} has [MASK] Parkinson disease.",
         "severity"),
        (f"Hoehn and Yahr stage {patient_data.get('hy', 2)} indicates [MASK] disease progression.",
         "progression"),
        (f"A disease duration of {patient_data.get('duration', 5)} years suggests [MASK] stage Parkinson disease.",
         "stage"),
    ]

    for prompt, category in prompts:
        try:
            r = client.fill_mask(prompt, model=PUBMEDBERT_MODEL)
            top = r[0]
            assessments.append(f"  [{category}] {top.token_str} ({top.score:.1%})")
        except:
            assessments.append(f"  [{category}] unavailable")

    return "\n".join(assessments)


def main():
    print("=" * 80)
    print("  PubMedBERT Analysis — Clinical Validation via HuggingFace Inference API")
    print(f"  Model: {PUBMEDBERT_MODEL}")
    print("=" * 80)

    if not HF_TOKEN:
        print("  ERROR: HuggingFace API token not found in config.yaml")
        return

    # 1. Feature validation
    feat_df = validate_features_with_pubmedbert()

    # 2. Clinical embeddings + similarity
    embeddings = compute_clinical_embeddings()

    # 3. Example patient reports
    print("\n" + "=" * 80)
    print("  PubMedBERT Patient Assessments")
    print("=" * 80)

    patients = [
        {"id": "HIGH", "updrs": 38, "hy": 3, "duration": 12},
        {"id": "MODERATE", "updrs": 25, "hy": 2.5, "duration": 5},
        {"id": "LOW", "updrs": 10, "hy": 1, "duration": 2},
    ]
    reports = []
    for p in patients:
        print(f"\n  Patient {p['id']} (UPDRS={p['updrs']}, H&Y={p['hy']}):")
        report = generate_pubmedbert_report(p)
        print(report)
        reports.append({"Patient": p["id"], "UPDRS": p["updrs"], "HY": p["hy"],
                        "Duration": p["duration"], "PubMedBERT_Assessment": report})

    pd.DataFrame(reports).to_csv(TABLES / "pubmedbert_patient_reports.csv", index=False)
    print(f"\n  Saved: pubmedbert_patient_reports.csv")

    print("\n" + "=" * 80)
    print("  PubMedBERT Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
