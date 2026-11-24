# Anaemia Prediction Research Prototype

This Flask app uses PyTorch to classify CBC inputs as anaemic or non-anaemic. Both manual entry and text-based PDF upload use the same saved PyTorch checkpoint and scaler. The model uses Gender, Hemoglobin, MCH, MCHC, and MCV as inputs. The result page shows the classification, a confidence score, and a rule-based risk index.

Manual inputs and PDF contents are sent to the hosted Flask server for processing. Use de-identified reports or synthetic data.

## PyTorch holdout evaluation

| Metric | Result |
| --- | ---: |
| Accuracy | 99.1% |
| Precision | 98.0% |
| Recall | 100% |
| F1-score | 99.0% |

The evaluation removed exact duplicate rows from the labeled dataset, kept the first occurrence, and used a stratified 80:20 split with `random_state=42`. The holdout contained 107 records. The positive class was anaemic (`Result=1`), and the sigmoid threshold was 0.5. The confusion matrix was `[[57, 1], [0, 49]]`, with actual classes as rows, predicted classes as columns, and class 0 before class 1.

These results describe one dataset holdout. Independent clinical validation has not been performed, and predictions do not establish a diagnosis.

## Historical SVM experiments

The [legacy notebook](Anaemia_XAI_Project/notebook/anaemia_xai.ipynb) preserves the SVM experiments and their original results. The website shows static SHAP and counterfactual examples from that work. Live PyTorch requests do not generate per-request SHAP or counterfactual explanations. The SVM implementation and artifacts are retained for historical reference and are not used as an automatic fallback.

## Project stack

Python, Flask, PyTorch, scikit-learn, pandas, PyMuPDF, HTML, CSS, and JavaScript. The historical notebook uses SHAP and Matplotlib.
