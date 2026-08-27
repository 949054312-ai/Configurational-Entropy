# Configurational Entropy

Code repository associated with the manuscript:

**Machine Learning-Driven Configurational Entropy as a Descriptor for Photovoltaic Performance in Multicomponent Organic Solar Cells**

## Repository contents

### `ML_model.py`
Trajectory-level machine-learning benchmarking using molecular-dynamics-derived descriptors, including regression-model comparison and CatBoost analysis.

### `ML_model_analysis.py`
Trajectory-level CatBoost model evaluation and interpretation, including five-fold cross-validation, feature importance, permutation analysis, and reduced-dimensional descriptor-space analysis.

### `ML_single_molecule_fixed_v4_3_diagnostic.py`
Monomer-level descriptor aggregation, feature-selection benchmarking, regression-model comparison, and leave-one-out cross-validation for configurational entropy prediction.

### `Pearson-63-features.py`
Pearson correlation analysis and visualization for the 63 trajectory-derived descriptors retained for downstream analysis.

### `Shap_analysis_modified_styled.py`
SHAP-based interpretation and visualization of the 63-descriptor CatBoost model.

### `energy_disorder_calculate.py`
Calculation and aggregation of blend-level energetic-disorder and structural-fluctuation descriptors from molecular-dynamics-derived data.

### `energy_disorder_analysis.py`
Statistical analysis and visualization of relationships among configurational entropy, energetic disorder, and structural-fluctuation descriptors.

### `predict_new_blends_original_ridge.py`
Direct prediction of new multicomponent formulations using the original fitted Ridge regression model with correlation-based feature selection. This script performs prediction only and does not retrain the model.

## Data availability

Source data supporting the analyses are provided with the article as Supporting Information and accompanying Source Data.

Several scripts require the corresponding input datasets or fitted model files specified within the scripts. These files should be placed in the working directory using the filenames expected by each script.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Notes

The scripts correspond to the machine-learning, descriptor-analysis, model-interpretation, and energetic-disorder analyses reported in the manuscript and Supporting Information.

Please refer to the manuscript and Supporting Information for complete computational procedures, descriptor definitions, data-processing protocols, model-evaluation procedures, and interpretation of the results.

## Citation

If you use this code, please cite the associated article after publication.
