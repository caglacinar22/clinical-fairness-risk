# Fairness-Aware Clinical Risk Prediction

A multimodal machine learning system for patient deterioration 
prediction with explicit fairness auditing and adversarial 
debiasing across demographic subgroups.

## Motivation
Clinical ML models trained on historical data often encode 
systemic biases — underperforming for women, elderly patients, 
or minority groups not well represented in training data. 
This project explicitly measures and corrects for those disparities.

## Approach
1. **Baseline model** — multimodal fusion of vitals (LSTM) and clinical notes (BERT)
2. **Fairness audit** — measure equalized odds, calibration parity, demographic parity across subgroups
3. **Adversarial debiasing** — penalize model for encoding protected attributes
4. **Tradeoff analysis** — fairness vs accuracy frontier

## Dataset
MIMIC-III (PhysioNet) — pending credentialed access

## Tech Stack
- PyTorch
- HuggingFace Transformers
- FairLearn
- SHAP
- scikit-learn

## Project Status
 In progress

## Author
Cagla CINAR — BSc Computer Engineering, Politecnico di Torino