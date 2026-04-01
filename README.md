# Fairness-Aware Clinical Risk Prediction

A multimodal machine learning system for patient deterioration 
prediction with explicit fairness auditing and adversarial 
debiasing across demographic subgroups.

## Motivation
Clinical ML models trained on historical data often encode systemic biases — underperforming for women
elderly patients,or minority groups not well represented in training data. 
This project explicitly measures and corrects for those disparities.

## Approach
1. **EDA** — explore data, visualize subgroup distributions
2. **Preprocessing** — clean, scale, split, preserve subgroup labels
3. **Baseline Model** — logistic regression, measure subgroup performance
4. **Fairness Audit** — demographic parity, equalized odds across sex and age
5. **Adversarial Debiasing** — ExponentiatedGradient with equalized odds constraint

## Key Results
| Metric | Baseline | Fair Model |
|--------|----------|------------|
| ROC-AUC | see results/ | see results/ |
| Demographic Parity Diff | see results/ | see results/ |
| Equalized Odds Diff | see results/ | see results/ |

## Dataset
MIMIC-III (PhysioNet) — currently using Heart Failure Clinical Records (UCI) as proxy

## Tech Stack
- PyTorch
- HuggingFace Transformers
- FairLearn
- SHAP
- scikit-learn

## Project Structure
```
├── data/
│   ├── raw/               # original data
│   └── processed/         # cleaned, split data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_fairness_audit.ipynb
│   └── 05_debiasing.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── fairness.py
│   └── train.py
└── results/
    └── figures/
```

## Author
Cagla CINAR — BSc Computer Engineering, Politecnico di Torino