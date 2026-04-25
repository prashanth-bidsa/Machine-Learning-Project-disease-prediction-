# Disease Prediction System
> Machine Learning based disease prediction using patient symptom data

**Name:** Prashanth P. | **USN:** 25MSDSR067
**Course:** Machine Learning Project | **Program:** M.Sc. Data Science and Analytics — B Section

---

## Problem Statement

Early diagnosis is critical in healthcare but doctors often face overlapping symptoms across thousands of records. This project builds an ML-based system that analyzes structured patient symptom data to predict disease categories with high accuracy and support clinical decision-making.

---

## Repository Structure

```
disease-prediction-system/
│
├── disease_prediction.py        # Main ML pipeline (preprocessing + training + evaluation)
├── disease_dataset.csv          # Dataset — 4,920 records, 132 symptoms, 41 diseases
├── Disease_Prediction_Report.docx  # Full project report
│
├── figures/
│   ├── fig1_model_comparison.png
│   ├── fig2_confusion_matrix.png
│   ├── fig3_feature_importance.png
│   ├── fig4_per_disease_recall.png
│   ├── fig5_disease_distribution.png
│   └── fig6_symptom_cooccurrence.png
│
└── README.md
```

---

## Dataset

| Attribute | Value |
|---|---|
| Source | [Kaggle — Disease Prediction Using ML](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| Records | 4,920 |
| Features | 132 binary symptom columns |
| Target Classes | 41 diseases |
| Missing Values | None |
| Class Balance | Perfectly balanced (120 records/disease) |
| Train / Test Split | 80% / 20% (stratified) |

---

## Models Trained

| Model | Accuracy | Recall | F1-Score |
|---|---|---|---|
| **SVM (Best)** | **99.70%** | **99.70%** | **99.69%** |
| Random Forest | 99.59% | 99.59% | 99.59% |
| Gradient Boosting | 98.48% | 98.48% | 98.47% |
| Decision Tree | 77.54% | 77.54% | 79.47% |
| Naive Bayes | 73.98% | 73.98% | 73.31% |

---

## Key Findings

- **SVM with RBF kernel** achieved the best accuracy of **99.70%** — ideal for high-dimensional binary feature spaces
- **Random Forest** provides feature importance for clinical interpretability
- Top predictive symptoms: `high_fever`, `vomiting`, `fatigue`, `loss_of_appetite`, `headache`, `itching`, `yellowish_skin`
- Jaundice-related symptoms form the most distinct cluster (near 100% recall)

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn pandas numpy matplotlib seaborn

# Create figures directory
mkdir figures

# Run the pipeline
python disease_prediction.py
```

---

## Tools & Libraries

`Python 3` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---
