# Spatial Neighborhood-aware Brain-layer Classification using Visium DLPFC Data

This project is a term-project workflow for classifying human dorsolateral prefrontal cortex (DLPFC) spatial transcriptomics spots into brain-layer labels using Visium data.

The project focuses on seven target classes:

- Layer1
- Layer2
- Layer3
- Layer4
- Layer5
- Layer6
- WM

The main goal is to test whether stable recurrent layer-associated genes and spatial neighborhood information improve brain-layer classification compared with expression-only models.

## Project Overview

The workflow includes:

1. Loading labelled Visium DLPFC spatial transcriptomics samples
2. Applying quality filtering and label cleaning
3. Performing exploratory spatial and clustering analysis
4. Identifying stable recurrent layer-associated genes
5. Training baseline models using highly variable genes
6. Training expression-only stable-gene models
7. Training neighborhood-aware seven-class models
8. Training a final Linear SVM model
9. Validating the final model on a limited labelled spatialDLPFC external subset
10. Preparing report-ready result tables and figures

## Datasets

### Internal Dataset

The main internal dataset contains 12 labelled Visium DLPFC tissue sections:

```text
151507, 151508, 151509, 151510,
151669, 151670, 151671, 151672,
151673, 151674, 151675, 151676
```

After filtering, the final internal training dataset contained:

```text
47,020 spots
7 target labels
```

### External Dataset

Limited external validation was performed using three manually labelled spatialDLPFC samples:

```text
Br6522_ant
Br6522_mid
Br8667_post
```

The external validation is limited because only these three samples had manual layer labels, and Br8667_post did not contain Layer1 or WM labels.

## Main Methods

The project compared multiple feature and model settings.

### Feature Modes

```text
HVG features
Stable recurrent gene features
Stable recurrent genes + spatial neighborhood features
```

### Models

```text
Logistic Regression
Random Forest
Linear SVM
```

### Neighborhood Settings

```text
k = 3
k = 5
k = 10
```

The final selected model used:

```text
Linear SVM
85 stable recurrent genes
k = 10 spatial neighborhood features
170 total features
```

## Main Results

The main internal result showed that adding spatial neighborhood features improved performance.

```text
Expression-only stable-gene Linear SVM:
Weighted F1 = 0.6768

Stable genes + k=10 neighborhood features:
Weighted F1 = 0.8196

Stable genes + k=10 neighborhood features + smoothing:
Weighted F1 = 0.8369
```

Limited external validation on the spatialDLPFC labelled subset achieved:

```text
Accuracy = 0.6798
Weighted F1 = 0.6789
Macro F1 = 0.6708
```

Layer5, Layer6, and WM showed stronger external transfer, while Layer2 and Layer4 remained the weakest classes.

## Code Files

### Python Scripts

| File | Purpose |
|---|---|
| `run_visium_h5ad_pipeline.py` | Loads and processes individual internal Visium DLPFC H5AD samples. |
| `run_visium_all12_comparison.py` | Summarizes QC, clustering, and comparison results across all 12 internal samples. |
| `run_all12_signature_gene_recurrence.py` | Finds stable recurrent signature genes across the internal samples. |
| `run_visium_ml_classification_hvg.py` | Runs expression-only baseline classification using highly variable genes. |
| `run_visium_ml_classification.py` | Runs expression-only baseline classification using stable recurrent genes. |
| `run_visium_ml_classification_7class_neighborhood.py` | Runs the main seven-class neighborhood-aware classification experiments. |
| `train_final_7class_model.py` | Trains the final seven-class Linear SVM model on all internal samples. |
| `validate_external_spatialDLPFC_model.py` | Validates the final saved model on the limited labelled spatialDLPFC external subset. |
| `prepare_final_model_report_package.py` | Prepares final result summaries, figures, and report-ready outputs. |

### R Scripts

| File | Purpose |
|---|---|
| `download_spatialDLPFC.R` | Downloads the spatialDLPFC data. |
| `export_spatialDLPFC_labelled_samples.R` | Exports labelled spatialDLPFC samples for external validation. |

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Main Python packages used include:

```text
scanpy
squidpy
pandas
numpy
matplotlib
scikit-learn
scipy
leidenalg
igraph
anndata
h5py
joblib
```

## Output Folders

The project generates outputs such as:

```text
all12_visium_comparison/
visium_hvg_classification/
visium_stable_gene_classification/
visium_7class_neighborhood_classification/
outputs/models/final_7class_model/
outputs/reports/
```

These folders contain figures, tables, model files, validation reports, and final report-ready summaries.

## Project Limitation

This project is designed as a term-project workflow, not a final production-level biological model. The main limitation is that external validation was performed on only three manually labelled spatialDLPFC samples. More labelled external DLPFC or cortical spatial transcriptomics datasets would be needed for stronger generalization claims.

## Final Model Summary

```text
Classifier: Linear SVM
Feature mode: Stable recurrent genes
Selected genes: 85
Neighborhood size: k = 10
Training spots: 47,020
Training features: 170
Target labels: Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, WM
```
