# NATURE-CARBON Canada Proof-of-Concept Python Simulation

## Overview

This repository-style proof of concept supports the proposal **NATURE-CARBON Canada: Nature-Based Carbon and Biodiversity Intelligence for Canadian Natural Climate Solutions**. The script demonstrates, in miniature, how an open hybrid mechanistic-AI platform could integrate ecosystem carbon, avoided conversion, restoration, improved management, biodiversity co-benefits, human well-being indicators, uncertainty, and decision-priority scoring into policy-relevant natural climate solution outputs.

The script is **not an operational carbon accounting system** and does not use real federal, provincial, territorial, Indigenous, satellite, inventory, or project-level monitoring data. Instead, it creates a scientifically structured synthetic dataset that mimics the logic of the proposed NATURE-CARBON Canada platform. This allows the full workflow to be demonstrated reproducibly without requiring access to national carbon inventories, wetland/peatland products, forest inventory systems, species-at-risk layers, protected-area datasets, or community-controlled data.

## Main script

```text
naturecarbon_poc.py
```

Running this script will generate a complete proof-of-concept analysis and save all outputs to:

```text
naturecarbon_poc_outputs/
```

## What the script demonstrates

The script demonstrates the following NATURE-CARBON Canada capabilities:

1. Creation of a harmonized spatiotemporal nature-carbon landscape database.
2. Simulation of Canadian ecozone and ecosystem differences across forests, wetlands, peatlands, grasslands, and riparian systems.
3. Simulation of carbon pools, including total carbon density, biomass carbon, and soil/peat carbon.
4. Simulation of land-use pressure, including agricultural, urban, and resource-development pressure.
5. Simulation of degradation, hydrological integrity, restoration readiness, monitoring quality, and stewardship context.
6. Simulation of climate and disturbance drivers, including drought, wetness, wildfire risk, pest risk, and temperature anomaly.
7. Estimation of avoided conversion, restoration, and improved-management scenarios.
8. Mechanistic reduced-form estimation of net GHG benefit using avoided emissions, carbon-stock gain, methane penalty, reversal risk, leakage risk, and implementation disturbance.
9. A pure AI baseline model using machine learning directly on environmental, carbon, climate, disturbance, biodiversity, and action features.
10. A hybrid mechanistic-AI residual model in which machine learning corrects the residual structure left by the mechanistic model.
11. Blocked validation by ecozone to test whether models generalize across ecological regions.
12. Bootstrap uncertainty intervals for hybrid net GHG benefit prediction.
13. Calculation of biodiversity co-benefit, human well-being co-benefit, monitoring readiness, permanence risk, leakage risk, and uncertainty indicators.
14. Transparent decision-priority scoring for natural climate solution portfolios.
15. Decision-weight sensitivity analysis to test whether priority rankings are robust to different policy preferences.
16. Accounting-parameter sensitivity analysis for methane, reversal risk, leakage, and carbon-gain assumptions.
17. Bootstrap parameter-identifiability analysis to show which simplified accounting terms are stable or unstable.
18. Export of reproducible CSV tables and publication-style figures.

## Required Python packages

The script uses standard scientific Python packages:

```bash
pip install numpy pandas matplotlib scikit-learn
```

Recommended Python version:

```text
Python 3.9 or newer
```

The script has no internet requirement and should run locally after the required packages are installed.

## How to run

Place the script in a working folder and run:

```bash
python naturecarbon_poc.py
```

In Jupyter Notebook or JupyterLab, you can run it from a notebook cell with:

```python
%run naturecarbon_poc.py
```

The script will print a console summary and create an output folder called:

```text
naturecarbon_poc_outputs
```

## Expected runtime

On a typical laptop, the script should usually finish in a few minutes or less. Runtime may vary depending on CPU speed, Python environment, and the configured number of regions, bootstrap models, and sensitivity samples.

## Output files

The script saves the following core CSV files:

`synthetic_landscape_database.csv` Full synthetic region-year dataset containing ecozone, ecosystem, carbon pool, land-use pressure, climate, disturbance, biodiversity, well-being, monitoring, and baseline GHG variables.

`scenario_results_raw.csv` Full region-year-action dataset before final hybrid uncertainty and priority scoring.

`scenario_results.csv` Main analysis table containing action-specific GHG outcomes, hybrid predictions, uncertainty intervals, co-benefits, risk indicators, and priority scores.

`model_performance.csv` Overall model-performance metrics for mechanistic-only, AI-only, and hybrid mechanistic-AI residual models.

`model_performance_by_fold.csv` Ecozone-blocked validation results by fold, including held-out ecozones and model metrics.

`ecosystem_action_summary.csv` Aggregated scenario summaries comparing mean predicted GHG benefit, biodiversity benefit, human well-being benefit, and uncertainty by ecosystem and action.

`top_priority_regions.csv` Top-ranked region-year-action combinations based on the transparent decision-priority score.

`sensitivity_weights.csv` Decision-weight sensitivity results showing how often each region-year-action appears in the top 5 percent under alternative weighting assumptions.

`sensitivity_parameters.csv` Accounting-parameter sensitivity results for methane penalty, reversal deduction, leakage deduction, and carbon-gain multiplier.

`parameter_sensitivity_correlations.csv` Correlations between accounting assumptions and the mean net GHG benefit.

`parameter_bootstrap_summary.csv` Bootstrap summary of simplified accounting-parameter estimates, including mean, standard deviation, 5th percentile, 95th percentile, and coefficient of variation.

`parameter_bootstrap_coefficients.csv` Bootstrap coefficient estimates from repeated simplified accounting-model fits.

`parameter_bootstrap_correlation.csv` Correlation matrix among bootstrap accounting-parameter estimates, useful for diagnosing identifiability and confounding.

The script also saves the following figures:

`fig01_model_rmse.png` RMSE comparison across mechanistic-only, AI-only, and hybrid models.

`fig02_hybrid_predicted_vs_true.png` Observed-versus-predicted plot for the hybrid mechanistic-AI model.

`fig03_scenario_outcomes_by_ecosystem.png` Mean predicted net GHG benefit by ecosystem and natural climate solution action.

`fig04_carbon_biodiversity_synergy.png` Carbon-benefit versus biodiversity-benefit trade-off and synergy space.

`fig05_priority_map.png` Synthetic map-like scatter plot highlighting the top 5 percent priority region-action combinations.

`fig06_uncertainty_by_ecosystem.png` Hybrid predictive uncertainty interval width by ecosystem type.

`fig07_weight_sensitivity.png` Relationship between base priority score and frequency of selection under alternative decision weights.

`fig08_parameter_sensitivity.png` Sensitivity of mean net GHG benefit to methane, reversal, leakage, and carbon-gain assumptions.

`fig09_parameter_identifiability.png` Bootstrap identifiability plot showing coefficient stability for simplified accounting terms.

## Model design

The script compares three modelling strategies.

### 1. Mechanistic-only model

The mechanistic model uses simplified process-based equations to estimate net GHG benefit from natural climate solution actions. It represents avoided conversion, carbon-stock gain, methane penalty, reversal risk, leakage risk, and implementation disturbance. This model is interpretable and aligned with carbon-accounting logic, but it is deliberately imperfect because real ecosystems contain unresolved nonlinearities, spatial heterogeneity, data gaps, hydrological uncertainty, and disturbance interactions.

### 2. Pure AI model

The pure AI model uses machine learning directly on the environmental, carbon, climate, land-use, disturbance, biodiversity, monitoring, and action features to predict net GHG benefit. This provides a flexible predictive benchmark, but it is less process-explicit and is not intended to replace transparent carbon accounting.

### 3. Hybrid mechanistic-AI residual model

The hybrid model first uses the mechanistic model to generate an interpretable baseline estimate. It then trains a machine-learning model on the residual error:

```text
True simulated net GHG benefit - Mechanistic prediction = residual structure
```

The final hybrid prediction is:

```text
Hybrid prediction = Mechanistic prediction + AI-predicted residual
```

This reflects the NATURE-CARBON Canada proposal philosophy: artificial intelligence is used as a scientifically constrained residual-correction and downscaling layer, while the mechanistic environmental model remains the backbone for interpretation, reporting, and policy credibility.

## Natural climate solution scenarios

The script evaluates three action families.

`protect` Protection / avoided conversion. This action estimates avoided emissions where carbon-rich ecosystems face land-use pressure and are not already protected. It includes additionality, leakage risk, reversal risk, and monitoring readiness.

`restore` Restoration. This action estimates carbon recovery in degraded ecosystems, with special attention to restoration readiness, hydrological integrity, biomass/soil recovery, and methane-sensitive wetland or peatland outcomes.

`manage` Improved management. This action estimates GHG benefits from reducing disturbance-related losses and improving carbon storage through management interventions, especially in forest, grassland, wetland, and riparian systems.

Each scenario produces a consistent decision-support vector containing:

- predicted net GHG benefit,
- avoided emissions,
- carbon-stock gain,
- methane penalty,
- biodiversity co-benefit,
- human well-being co-benefit,
- monitoring readiness,
- permanence/reversal risk,
- leakage risk,
- uncertainty score,
- priority score.

## Validation design

The script uses **blocked validation by ecozone**. This means model training and testing are separated by ecological regions rather than by random rows. This is more realistic than random splitting because a national decision-support platform must generalize across ecozones, ecosystem types, disturbance regimes, carbon-pool structures, and monitoring contexts.

The validation reports:

- R²,
- RMSE,
- MAE,
- bias,
- held-out ecozones for each fold.

A successful run should usually show that the hybrid mechanistic-AI residual model improves on the mechanistic-only model while preserving interpretability.

## Uncertainty, sensitivity, and identifiability

The script includes three uncertainty-oriented modules.

### Bootstrap predictive uncertainty

Multiple bootstrap residual models are trained to generate a distribution of hybrid predictions. The output includes 5th and 95th percentile intervals, interval width, and empirical interval coverage.

### Decision-weight sensitivity

The decision-priority score depends on transparent weights for GHG benefit, biodiversity benefit, human well-being benefit, monitoring readiness, uncertainty, and permanence risk. The script repeatedly samples alternative weight combinations and records how often each region-year-action remains in the top 5 percent. This helps distinguish robust priorities from weight-sensitive priorities.

### Accounting-parameter sensitivity and identifiability

The script perturbs methane penalty, reversal-risk deduction, leakage deduction, and carbon-gain assumptions to test how strongly accounting choices affect the mean net GHG benefit. It also fits simplified accounting equations repeatedly using bootstrap resampling to examine whether key accounting terms are stable or confounded.

## How to interpret the outputs

The proof-of-concept results should be interpreted as a feasibility demonstration rather than empirical evidence about actual Canadian natural climate solution outcomes. The synthetic data are designed to show that the proposed modelling architecture can be implemented end-to-end.

A successful run demonstrates that:

1. A harmonized nature-carbon data structure can be created.
2. Carbon pools, land-use pressure, climate disturbance, biodiversity, well-being, monitoring quality, and stewardship context can be represented together.
3. Avoided conversion, restoration, and improved management can be compared within one common accounting framework.
4. Reduced-form mechanistic equations can generate interpretable GHG predictions.
5. Machine learning can be used as a residual-correction layer rather than an opaque replacement for process modelling.
6. Blocked validation can compare mechanistic-only, AI-only, and hybrid models.
7. Uncertainty intervals can be attached to predicted GHG outcomes.
8. Biodiversity and human well-being co-benefits can be included before final prioritization rather than added after carbon ranking.
9. Priority rankings can be stress-tested under alternative decision weights.
10. Parameter sensitivity and identifiability can be reported to reduce overclaiming and improve transparency.

## Reproducibility

The script uses a fixed random seed defined in the configuration section:

```python
random_seed: int = 42
```

Changing the seed will generate a different synthetic dataset but should preserve the same general workflow and modelling logic.

## Configuration

At the top of the script, the `Config` dataclass controls the main simulation settings:

```python
@dataclass
class Config:
    random_seed: int = 42
    n_regions: int = 900
    n_years: int = 8
    start_year: int = 2023
    output_dir: str = "naturecarbon_poc_outputs"
    n_bootstrap: int = 40
    n_weight_samples: int = 300
    n_parameter_samples: int = 300
```

You can modify these values to increase or decrease the number of synthetic regions, change the simulated time window, rename the output directory, adjust the number of bootstrap uncertainty models, or change the number of sensitivity-analysis samples.

## Important limitations

This script intentionally uses synthetic data. Therefore:

- it should not be used to make real policy claims,
- it should not be interpreted as a Canadian GHG inventory,
- it does not estimate real regional carbon storage or avoided emissions,
- it does not replace Canada’s forest carbon reporting system, peatland models, wetland inventories, protected-area databases, species-at-risk datasets, satellite products, or project-level monitoring,
- it does not incorporate Indigenous or community-controlled data,
- it is intended only to demonstrate scientific and computational feasibility.

The next development step would be to replace synthetic inputs with real harmonized data streams, including forest carbon products, wetland and peatland layers, land-cover change data, disturbance datasets, protected-area and species-at-risk layers, climate projections, remote-sensing products, and partner-approved place-based information where appropriate. The platform should then be validated against observed field, inventory, remote-sensing, and project-level monitoring data before being used for policy or reporting decisions.
