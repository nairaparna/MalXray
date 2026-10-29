# MalXray – Next-Generation Malware Detection

**Project Title:** Next-Generation Malware Detection: Fusion of Static and Dynamic Analysis for Family Classification

**Student:** Aparna Pradeep Nair  
**Guide:** Dr. Anupama Jawale  
**University:** University of Mumbai – Department of Information Technology

## About the Project
MalXray is a hybrid malware detection and family classification system that combines:
- Static analysis (PE headers, entropy, imported APIs)
- Dynamic analysis (runtime behaviour)
- Machine Learning (XGBoost)

## Features
- Static + Dynamic feature fusion
- Malware family classification
- Desktop GUI + CLI
- VirusTotal cross-check
- PDF report generation

## How to Run
```bash
pip install -r requirements.txt
python generate_training_data.py --mode both
python train_model.py
python MalXray(advanced).py
