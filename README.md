# GE-79 Cognitive-Status Classification

## Diabetes, Cerebrovascular Biomarkers, and Responsible Machine Learning

**AI4ALL Ignite 2026 · Group 6C**

This supervised machine-learning project investigates whether clinical, cardiovascular, glycemic, inflammatory, cerebrovascular, gait, anthropometric, and MRI-derived biomarkers from the GE-79 PhysioNet cohort can help classify cognitive status in older adults with Type 2 Diabetes. The project compares three classifiers while treating responsible AI as part of the technical pipeline—not as a decorative paragraph added after the model is finished.

> **Research and education use only:** This project is exploratory. It has not been externally or clinically validated and must not be used for diagnosis, screening, triage, treatment, prognosis, or clinical decision-making.

## Project Links

| Resource | Purpose | Link |
|---|---|---|
| Google Colab | Reproducible notebook and model workflow | [Open Google Colab](https://tinyurl.com/AI4ALL-Collab) |
| Interactive prototype | Public-facing project demonstration | [Open the Group 6C prototype](https://tinyurl.com/AI4ALL-Group6C-Prototype) |
| GitHub Pages poster | Visual project poster and public showcase | [View the GitHub Pages poster](https://tinyurl.com/AI4ALL-GitHubPage) |
| Streamlit model visualizations | Interactive model results and data visualizations | [Open the Streamlit application](https://tinyurl.com/AI4ALL-Streamlit-App) |
| Final presentation | Group 6C project presentation | [View the presentation](https://tinyurl.com/AI4ALL-Group6C-Presentation) |
| GitHub repository | Source code, documentation, outputs, and project artifacts | [View the repository](https://github.com/elizabethhannan/AI4ALL-Diabetes-PRIVATE-ML) |

### Direct Streamlit Model Pages

- [Model 0 — Feature Selection](https://ai4all-diabetes-ml-model-0-features.streamlit.app/)
- [Model 1 — Logistic Regression](https://ai4all-diabetes-app-ml-model-1-logistic-regression.streamlit.app/)
- [Model 2 — Decision Tree](https://ai4all-diabetes-app-ml-model-2-decision-tree.streamlit.app/)
- [Model 3 — Random Forest](https://ai4all-diabetes-app-ml-model-3-random-forest.streamlit.app/)
- [Bias and Responsible AI Reports](https://i4all-diabetes-ml-bias-report.streamlit.app/)

## Research Question

Can clinical, cardiovascular, glycemic, inflammatory, cerebrovascular, and MRI-derived biomarkers from the GE-79 cohort be used to classify mild cognitive impairment in older adults with Type 2 Diabetes, and how can the machine-learning workflow reduce or document bias during model development?

## Why This Project Matters

Cognitive decline can be difficult to identify early, particularly when health conditions overlap. Type 2 Diabetes is associated with vascular, inflammatory, and metabolic changes that may also relate to cognitive health. This project tests whether those measurable biomarkers contain a meaningful classification signal.

The purpose is not to create a clinical shortcut. It is to evaluate the data honestly, compare models consistently, make errors visible, and document where the evidence is too limited to support stronger claims. In healthcare machine learning, a polished accuracy score can hide a rather inconvenient truth: the model may still be missing the people whose condition matters most.

## Dataset

| Item | Description |
|---|---|
| Primary source | GE-79 / PhysioNet, *Cerebromicrovascular Disease in Elderly with Diabetes*, version 1.0.1 |
| Cleaned modeling file | `data/GE79_MASTER_DATASET_V1.csv` |
| Participants | 75 |
| Variables | 46 columns before final feature locking |
| Target | `cognitive_status_label` |
| Class distribution | 55 Normal / No Impairment; 20 Mild Impairment / Impaired |
| Supporting dataset | GE-75, reserved for future external-validation planning and excluded from the current GE-79 pipeline |
| Dataset documentation | `GE79_DATASET_CARD.md` |

The project uses a locked set of 14 biomarkers selected during Model 0 feature selection. That same predictor set is used across the three classifiers so that model comparisons reflect algorithmic differences rather than moving the goalposts between experiments. Exact dataset column names and feature explanations are retained in the feature-selection outputs, source code, and `docs/` materials.

### Data Governance and Privacy

- Direct identifiers were excluded from the modeling workflow.
- Target-derived and post-outcome variables were excluded from predictors to reduce leakage risk.
- The cleaned modeling file, dataset card, and data dictionary document the variables used in analysis.
- Raw or restricted data should not be republished outside the permissions and terms established by PhysioNet.
- The GE-75 reference cohort is not merged into the current training data.

## Machine-Learning Workflow

1. Review the GE-79 dataset, target definition, class balance, missingness, and candidate biomarkers.
2. Remove identifiers and exclude variables that could reveal or directly encode the target.
3. Use Model 0 to select and lock the shared 14-feature predictor set.
4. Place preprocessing inside scikit-learn pipelines so transformations are learned within the training folds.
5. Train Logistic Regression, Decision Tree, and Random Forest classifiers using the same inputs.
6. Evaluate all classifiers with five-fold stratified cross-validation.
7. Compare accuracy with macro F1, impaired recall, confusion matrices, ROC-AUC, and PR-AUC.
8. Use SHAP and feature-importance analysis to examine model behavior.
9. Audit fairness, reporting quality, governance, and risk using the responsible-AI framework set.
10. Document limitations, intended use, prohibited use, and future validation requirements.

### Preprocessing and Leakage Prevention

Preprocessing is performed inside the modeling pipelines rather than on the entire dataset before cross-validation. This separation reduces the risk that information from a held-out fold influences training. Missing-data handling and required transformations are applied consistently across classifiers, while the target and target-derived variables remain outside the predictor matrix. The repository code and Colab notebook are the controlling sources for the exact transformations.

## Models, Inputs, and Outputs

| Model | Algorithm | Inputs | Output | Strength | Limitation |
|---|---|---|---|---|---|
| Model 0 | Random Forest feature selection | Candidate GE-79 biomarkers | Locked 14-feature set | Reduces noisy predictors and standardizes downstream inputs | Feature importance may be unstable in a small sample |
| Model 1 | Logistic Regression | Locked 14-feature set | Class probability and predicted label | Interpretable linear baseline | Linear assumptions may miss nonlinear relationships |
| Model 2 | Decision Tree | Locked 14-feature set | Predicted cognitive-status label | Produces understandable decision rules | Can overfit a small dataset |
| Model 3 | Random Forest | Locked 14-feature set | Class probability, predicted label, and feature importance | Provides ensemble stability and the strongest aggregate discrimination | Less transparent than a single tree and produced low impaired recall |

## Evaluation Method

Because the dataset contains 55 participants without impairment and only 20 participants with impairment, accuracy cannot be treated as sufficient evidence of performance. Each classifier therefore uses the same five-fold stratified cross-validation design and is evaluated using:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Impaired-class recall
- Confusion matrix
- ROC-AUC
- PR-AUC

Macro F1 and impaired recall receive particular attention because they expose performance on the smaller impaired class. PR-AUC is also useful in this imbalanced setting because it focuses on the relationship between precision and recall for the positive class.

## Model Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Impaired Recall | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.547 | 0.525 | 0.532 | 0.508 | 0.500 | 0.534 | 0.369 |
| Decision Tree | 0.627 | 0.583 | 0.602 | 0.580 | **0.550** | 0.639 | 0.370 |
| Random Forest | **0.747** | **0.664** | 0.589 | **0.594** | 0.250 | **0.648** | **0.441** |

### Model Comparison and Selection

Random Forest produced the highest accuracy, macro F1, ROC-AUC, and PR-AUC. It is therefore the strongest model by several aggregate measures. That result is not the whole story: its impaired recall was only 0.250, meaning it missed many participants in the minority impaired group.

Decision Tree achieved the strongest impaired recall at 0.550, while Logistic Regression reached 0.500. The comparison does not justify presenting Random Forest as clinically superior. Instead, it demonstrates why model selection in a health-related setting must weigh error costs, minority-class performance, interpretability, and safety—not merely whichever number happens to be wearing the largest hat.

## Explainability

The project uses Random Forest feature importance and SHAP to examine how the final model arrives at its predictions. Supporting outputs include:

- SHAP global feature importance
- SHAP summary and beeswarm plots
- Individual SHAP waterfall explanations
- Random Forest feature-importance visualizations
- Decision Tree structure and decision-path visualization

These methods improve transparency, but they do not establish causality or clinical validity. A feature that influences a model prediction is not automatically a biological cause of cognitive impairment.

## Bias and Responsible AI

AI/ML systems can amplify bias when teams treat data as neutral, optimize only for headline accuracy, ignore missingness, fail to inspect subgroup behavior, or use a model outside the population represented in development. Those risks are substantial here because the GE-79 dataset is small, imbalanced, geographically limited, and drawn from a single cohort.

Risk controls used in this project include:

- Removal of direct identifiers
- Exclusion of target-derived leakage features
- Consistent preprocessing across classifiers
- A shared locked feature set
- Five-fold stratified cross-validation
- Reporting of macro F1 and impaired-class recall
- Confusion-matrix, ROC-AUC, and PR-AUC analysis
- SHAP-supported model transparency
- Missing-data and dataset documentation
- VerifyWise fairness screening
- NIST AI RMF risk-management assessment
- OECD AI Principles mapping
- TRIPOD+AI reporting review
- PROBAST-AI risk-of-bias review
- Dataset Card and Model Card documentation

These controls make weaknesses more visible; they do not erase bias or convert an exploratory model into a safe clinical system.

## Potential Benefits, Harms, and Error Costs

### Potential Benefits

- Tests whether routinely collected biomarkers contain a reproducible cognitive-status signal.
- Provides a foundation for larger studies of cognitive decline in people with Type 2 Diabetes.
- Demonstrates model evaluation that looks beyond accuracy.
- Makes limitations and responsible-AI decisions visible to reviewers and future contributors.

### Potential Harms

- A false negative may classify an impaired participant as having no impairment, potentially delaying further evaluation in an inappropriate real-world use.
- A false positive may create unnecessary concern or lead to unnecessary follow-up.
- Limited subgroup representation may conceal uneven performance.
- Explanations may be mistaken for causal medical conclusions.
- Strong-looking aggregate metrics may encourage use beyond the evidence.

## Limitations

- Only 75 participants are included.
- The target classes are imbalanced: 55 without impairment and 20 with impairment.
- The dataset represents a limited, single-cohort population.
- Subgroup sample sizes are insufficient to establish fairness.
- Labels and missing data may introduce uncertainty.
- Cross-validation estimates internal performance but does not replace external validation.
- No prospective, multi-center, or clinical-impact validation has been completed.
- SHAP and feature importance describe model behavior; they do not demonstrate causal relationships.
- The reported results are exploratory and must not be generalized to clinical populations without additional evidence.

## Visualizations and Project Materials

The repository and linked applications include:

- [Interactive Group 6C prototype](https://tinyurl.com/AI4ALL-Group6C-Prototype)
- [Streamlit model data visualizations](https://tinyurl.com/AI4ALL-Streamlit-App)

- Feature-selection and target-distribution visualizations
- Interactive ECharts and Streamlit visualizations for Models 0–3
- Confusion matrices for Models 1–3
- ROC and precision-recall curves
- Decision Tree diagram
- Random Forest feature importance
- SHAP global, summary, beeswarm, and waterfall plots
- Dataset Card and Model Card
- VerifyWise, OECD, NIST, TRIPOD+AI, and PROBAST-AI summaries
- Presentation-export images and the public GitHub Pages poster

## Responsible-AI Documentation

The `AI4ALL_ML-Diabetes_Ver_1_A/bias_reports/` directory and the linked [Streamlit application](https://tinyurl.com/AI4ALL-Streamlit-App) contain:

- NIST AI Risk Management Framework assessment
- OECD AI Principles mapping
- VerifyWise fairness assessment
- Fairlearn future-work plan
- TRIPOD+AI checklist
- PROBAST-AI risk-of-bias report
- GE-79 Dataset Card
- GE-79 Model Card
- SHAP explainability report

## Reproducing the Analysis

### Google Colab

1. Open the [Group 6C Google Colab notebook](https://tinyurl.com/AI4ALL-Collab).
2. Save a personal copy if editing is required.
3. Confirm that the approved GE-79 data file is available at the path specified in the notebook.
4. Run the notebook sections in order: data review, preprocessing, feature selection, model training, evaluation, and visualization.
5. Compare the reproduced metrics with the results table in this README.

### Local Repository

```bash
git clone https://github.com/elizabethhannan/AI4ALL-Diabetes-PRIVATE-ML.git
cd AI4ALL-Diabetes-PRIVATE-ML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the model scripts in `src/` for training and evaluation. Streamlit entry files and shared visualization components are located in `app/`. Exact application entry points are documented alongside those files.

## Quality-Assurance Checks

The following checks support reproducibility and acceptance review:

- All supervised models use the same locked feature set.
- Preprocessing remains inside the cross-validation pipeline.
- Stratified folds preserve the target-class distribution as closely as possible.
- Predictors are checked for direct target leakage.
- Model tables, exported figures, and Streamlit displays are compared for consistency.
- Accuracy is never reported without minority-class and threshold-sensitive metrics.
- Claims remain consistent with the Dataset Card, Model Card, and risk-of-bias documentation.
- The research-only disclaimer appears in repository and public-facing materials.

## Repository Structure

```text
AI4ALL-Diabetes-PRIVATE-ML/
├── README.md
├── requirements.txt
├── app/                                     # Streamlit apps and shared visualization components
├── data/                                    # Cleaned GE-79 modeling data and documentation
├── docs/                                    # Feature explanations and supporting documentation
├── models/                                  # Saved model artifacts, when included
├── notebooks/                               # Colab/Jupyter analysis notebooks, when included
├── outputs/                                 # Metrics, figures, SHAP outputs, and exports
├── src/                                     # Feature-selection, training, and evaluation scripts
├── presentation/                            # Presentation and poster materials, when included
└── AI4ALL_ML-Diabetes_Ver_1_A/
    └── bias_reports/                        # Fairness, governance, model, dataset, and bias reports
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- SHAP
- Streamlit
- streamlit-echarts
- JavaScript (prototype)
- Google Colab
- Git and GitHub

## Future Work

| Priority | Next Step | Intended Output |
|---|---|---|
| 1 | Save out-of-fold predictions by participant | Versioned CSVs for Models 1–3 containing predicted class, probability, true label, and approved subgroup fields |
| 2 | Run Fairlearn MetricFrame analysis | Subgroup accuracy, recall, selection rate, false-positive rate, and false-negative rate |
| 3 | Evaluate GE-75 or another compatible cohort | Data-dictionary comparison, harmonization plan, and external-validation feasibility memo |
| 4 | Expand to larger and more diverse multi-center cohorts | More credible estimates of generalizability and subgroup performance |
| 5 | Reassess model-selection priorities | Written recommendation balancing impaired recall, PR-AUC, calibration, interpretability, and safety |
| 6 | Pair SHAP findings with longitudinal and causal analysis | Clear separation between predictive association and causal interpretation |

## Citations and Data Sources

1. Anthropic. (2026). *Claude Opus 4* [Large language model]. https://claude.ai/
2. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324
3. Cantú-Brito, C., Mimenza-Alvarado, A., & Sánchez-Hernández, J. J. (2010). Diabetes mellitus y el envejecimiento como factor de riesgo de enfermedad vascular cerebral: Epidemiología, fisiopatología y prevención [Diabetes mellitus and aging as a risk factor for cerebral vascular disease: Epidemiology, pathophysiology and prevention]. *Revista de Investigación Clínica, 62*(4), 333–342.
4. Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological), 20*(2), 215–242.
5. Imahori, Y., Vetrano, D. L., Ljungman, P., & Qiu, C. (2021). Electrocardiographic predictors of cognitive decline and dementia: A systematic review. *Journal of Alzheimer's Disease, 84*(3), 1303–1322. https://doi.org/10.3233/JAD-210606
6. Li, F., Zhao, S., Wu, T., Yang, S., Duan, Y., Sun, J., Zhu, W., Zhai, B., Yu, C., Chen, S., Zhang, Z., Tang, W., & Bi, Y. (2025). Predicting mild cognitive impairment in type 2 diabetes: A machine learning approach. *Journal of Diabetes Research, 2025*, 7304414. https://doi.org/10.1155/jdr/7304414
7. Novak, V., Last, D., Alsop, D. C., Abduljalil, A. M., Hu, K., Lepicovsky, L., Cavallerano, J., & Lipsitz, L. A. (2006). Cerebral blood flow velocity and periventricular white matter hyperintensities in type 2 diabetes. *Diabetes Care, 29*(7), 1529–1534. https://doi.org/10.2337/dc06-0261
8. Novak, V., Zhao, P., Manor, B., Sejdic, E., Alsop, D., Abduljalil, A., Roberson, P. K., Munshi, M., & Novak, P. (2011). Adhesion molecules, altered vasoreactivity, and brain atrophy in type 2 diabetes. *Diabetes Care, 34*(11), 2438–2441. https://doi.org/10.2337/dc11-0969
9. Novak, V., & Quispe, R. (2022). *Cerebromicrovascular disease in elderly with diabetes* (Version 1.0.1) [Data set]. PhysioNet. https://doi.org/10.13026/00bm-0x81
10. Shakibajahromi, B., Dolui, S., Brown, C. A., Taghvaei, M., Sadaghiani, S., Khandelwal, P., Das, S., Yushkevich, P. A., Nasrallah, I. M., Xie, S. X., Wolk, D. A., & Detre, J. A. (2025). Clinical correlates of perfusion and diffusion MRI metrics in cognitively unimpaired older adults. *Journal of Cerebral Blood Flow & Metabolism, 45*, 1024–1039. https://doi.org/10.1177/0271678X251389043
11. Verghese, J., Wang, C., Lipton, R. B., Holtzer, R., & Xue, X. (2007). Quantitative gait dysfunction and risk of cognitive decline and dementia. *Journal of Neurology, Neurosurgery & Psychiatry, 78*(9), 929–935. https://doi.org/10.1136/jnnp.2006.106914
12. Xia, S., Liu, Q., Yang, Z., Huang, J., Gong, Q., & collaborators. (2026). Integrated cortical-cognitive signatures identified by machine learning enable early detection of MCI in type 2 diabetes. *Scientific Reports, 16*, 44608. https://doi.org/10.1038/s41598-026-44608-z
13. Xu, Z., Zhao, L., Yin, L., Liu, Y., Ren, Y., Yang, G., Wu, J., Gu, F., Sun, X., Yang, H., Peng, T., Hu, J., Wang, X., Pang, M., Dai, Q., & Zhang, G. (2022). MRI-based machine learning model: A potential modality for predicting cognitive dysfunction in patients with type 2 diabetes mellitus. *Frontiers in Bioengineering and Biotechnology, 10*, 1082794. https://doi.org/10.3389/fbioe.2022.1082794
14. Novak, V., Quispe, R., & Saunders, C. (2022). *Cerebral perfusion and cognitive decline in type 2 diabetes* (Version 1.0.1) [Data set]. PhysioNet. https://doi.org/10.13026/whjz-e968
15. Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. *Nature Health*. https://doi.org/10.1038/s44360-026-00096-z

## Authors and Contributors

Developed by Group 6C for AI4ALL Ignite 2026:

- Elizabeth Hannan — Models 1 and 3, integration, documentation, and public project materials ([Linkedin](https://www.linkedin.com/in/elizabethhanna))
- Agastyya Kala — Model 2, project development, and presentation ([LinkedIn](https://tinyurl.com/Linkedin-Agastyya-Kala))

## Acknowledgments

The team acknowledges AI4ALL Ignite for project guidance, PhysioNet and the original investigators for making the research dataset available, and the maintainers of the open-source tools used throughout the workflow.

## Use and Attribution

****All Rights Reserved
No copying, modification, distribution, deployment, or reuse without prior written permission
Only Elizabeth Hannan may grant permission
Third-party datasets and libraries retain their original licenses********
---

**AI4ALL Ignite 2026 · Group 6C · GE-79 Cognitive-Status Classification**
