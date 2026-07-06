"""
NATURE-CARBON Canada proof-of-concept simulation

This script creates scientifically structured proof-of-concept for NATURE-CARBON Canada.

Author: Mevan Rajakaruna, Harshana Rajakaruna
"""

from __future__ import annotations

import warnings
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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


CFG = Config()
rng = np.random.default_rng(CFG.random_seed)
OUT = Path(CFG.output_dir)
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_scale(x):
    x = np.asarray(x, dtype=float)
    denom = np.nanmax(x) - np.nanmin(x)
    if denom < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / denom


def savefig(name):
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close()


def print_header(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ---------------------------------------------------------------------
# 1. Synthetic Canadian nature-carbon landscape database
# ---------------------------------------------------------------------

def simulate_landscape_database(cfg: Config) -> pd.DataFrame:
    """
    Create a synthetic landscape database. Each row is a region-year unit.
    The variables are designed to resemble the domains in the proposal:
    ecosystem type, carbon pools, climate, disturbance, land-use pressure,
    restoration readiness, biodiversity, human well-being, governance context,
    and monitoring quality.
    """

    ecozones = [
        "Boreal Shield", "Boreal Plains", "Prairies", "Taiga Plains",
        "Atlantic Maritime", "Pacific Maritime", "Hudson Plains",
        "Mixedwood Plains", "Montane Cordillera"
    ]

    ecosystems = ["forest", "wetland", "peatland", "grassland", "riparian"]

    region_id = np.arange(cfg.n_regions)

    ecozone = rng.choice(ecozones, size=cfg.n_regions,
                         p=np.array([0.18, 0.13, 0.12, 0.10, 0.10, 0.10, 0.10, 0.10, 0.07]))

    # Ecosystem probabilities differ by ecozone to create structured heterogeneity.
    ecosystem = []
    for ez in ecozone:
        if ez in ["Hudson Plains", "Taiga Plains"]:
            p = [0.20, 0.22, 0.45, 0.04, 0.09]
        elif ez == "Prairies":
            p = [0.05, 0.15, 0.04, 0.64, 0.12]
        elif ez in ["Pacific Maritime", "Montane Cordillera"]:
            p = [0.62, 0.12, 0.08, 0.05, 0.13]
        elif ez == "Mixedwood Plains":
            p = [0.35, 0.20, 0.06, 0.15, 0.24]
        else:
            p = [0.52, 0.18, 0.12, 0.08, 0.10]
        ecosystem.append(rng.choice(ecosystems, p=p))
    ecosystem = np.array(ecosystem)

    # Approximate spatial coordinates in a projected synthetic space.
    x = rng.uniform(0, 100, cfg.n_regions)
    y = rng.uniform(0, 100, cfg.n_regions)

    # Latitudinal and coastal gradients.
    latitude_proxy = safe_scale(y)
    coastal_proxy = np.exp(-((x - 15) ** 2) / (2 * 16 ** 2)) + 0.7 * np.exp(-((x - 85) ** 2) / (2 * 18 ** 2))
    coastal_proxy = safe_scale(coastal_proxy)

    # Static landscape attributes.
    area_km2 = rng.lognormal(mean=2.7, sigma=0.55, size=cfg.n_regions)
    protection_status = rng.binomial(1, p=0.18 + 0.20 * (ecosystem == "peatland") + 0.10 * (ecosystem == "wetland"))
    protection_status = np.clip(protection_status, 0, 1)

    indigenous_stewardship_context = rng.binomial(
        1,
        p=np.clip(0.12 + 0.18 * (ecozone == "Taiga Plains") + 0.20 * (ecozone == "Hudson Plains")
                  + 0.09 * (ecosystem == "peatland"), 0, 0.65)
    )

    # Baseline carbon density by ecosystem, Mg CO2e per ha equivalent.
    carbon_mu = {
        "forest": 285,
        "wetland": 245,
        "peatland": 720,
        "grassland": 145,
        "riparian": 230,
    }
    carbon_sd = {
        "forest": 55,
        "wetland": 65,
        "peatland": 150,
        "grassland": 35,
        "riparian": 50,
    }

    carbon_density = np.array([
        np.maximum(30, rng.normal(carbon_mu[e], carbon_sd[e])) for e in ecosystem
    ])

    soil_fraction = np.where(ecosystem == "peatland", rng.uniform(0.70, 0.92, cfg.n_regions),
                      np.where(ecosystem == "grassland", rng.uniform(0.55, 0.80, cfg.n_regions),
                      np.where(ecosystem == "wetland", rng.uniform(0.45, 0.75, cfg.n_regions),
                               rng.uniform(0.25, 0.60, cfg.n_regions))))

    biomass_carbon = carbon_density * (1 - soil_fraction)
    soil_peat_carbon = carbon_density * soil_fraction

    # Degradation and land-use pressure.
    agriculture_pressure = safe_scale(1.2 * (ecozone == "Prairies").astype(float)
                                      + 0.9 * (ecozone == "Mixedwood Plains").astype(float)
                                      + 0.4 * rng.normal(size=cfg.n_regions)
                                      + 0.5 * (ecosystem == "grassland"))
    urban_pressure = safe_scale(0.9 * (ecozone == "Mixedwood Plains").astype(float)
                                + 0.5 * (ecozone == "Atlantic Maritime").astype(float)
                                + 0.5 * rng.normal(size=cfg.n_regions)
                                + 0.35 * coastal_proxy)
    resource_pressure = safe_scale(0.8 * (ecozone == "Boreal Plains").astype(float)
                                   + 0.8 * (ecozone == "Montane Cordillera").astype(float)
                                   + 0.6 * (ecozone == "Taiga Plains").astype(float)
                                   + 0.45 * rng.normal(size=cfg.n_regions))

    degradation = safe_scale(0.40 * agriculture_pressure + 0.25 * resource_pressure
                             + 0.20 * urban_pressure + 0.25 * rng.random(cfg.n_regions)
                             - 0.20 * protection_status)

    hydrology_integrity = np.clip(1 - degradation + rng.normal(0, 0.10, cfg.n_regions), 0, 1)
    monitoring_quality = np.clip(0.25 + 0.35 * protection_status + 0.20 * coastal_proxy
                                 + 0.15 * rng.random(cfg.n_regions), 0, 1)

    restoration_readiness = np.clip(0.25 + 0.45 * degradation + 0.20 * monitoring_quality
                                    + 0.15 * indigenous_stewardship_context
                                    - 0.20 * urban_pressure
                                    + rng.normal(0, 0.08, cfg.n_regions), 0, 1)

    # Biodiversity and well-being layers.
    sar_overlap = np.clip(0.20 + 0.22 * (ecosystem == "wetland") + 0.18 * (ecosystem == "grassland")
                          + 0.14 * (ecosystem == "riparian") + 0.15 * coastal_proxy
                          + rng.normal(0, 0.12, cfg.n_regions), 0, 1)

    connectivity = np.clip(0.55 + 0.15 * protection_status + 0.15 * (ecosystem == "riparian")
                           - 0.25 * urban_pressure - 0.20 * agriculture_pressure
                           + rng.normal(0, 0.10, cfg.n_regions), 0, 1)

    flood_exposure = np.clip(0.12 + 0.45 * (ecosystem == "wetland") + 0.38 * (ecosystem == "riparian")
                             + 0.18 * urban_pressure + rng.normal(0, 0.09, cfg.n_regions), 0, 1)

    heat_exposure = np.clip(0.15 + 0.45 * urban_pressure + 0.18 * agriculture_pressure
                            - 0.15 * coastal_proxy + rng.normal(0, 0.09, cfg.n_regions), 0, 1)

    community_vulnerability = np.clip(0.20 + 0.25 * flood_exposure + 0.20 * heat_exposure
                                      + 0.12 * indigenous_stewardship_context
                                      + rng.normal(0, 0.10, cfg.n_regions), 0, 1)

    rows = []
    years = np.arange(cfg.start_year, cfg.start_year + cfg.n_years)

    for year in years:
        t = year - cfg.start_year

        # Climate changes over time and varies by latitude/coastal context.
        temp_anomaly = 0.05 * t + rng.normal(0, 0.25, cfg.n_regions)
        drought_index = np.clip(0.25 + 0.20 * agriculture_pressure + 0.25 * temp_anomaly
                                + 0.12 * (ecozone == "Prairies") + rng.normal(0, 0.12, cfg.n_regions), 0, 1)
        wetness_index = np.clip(0.50 + 0.20 * coastal_proxy + 0.20 * (ecosystem == "wetland")
                                + 0.25 * (ecosystem == "peatland") - 0.25 * drought_index
                                + rng.normal(0, 0.10, cfg.n_regions), 0, 1)

        wildfire_risk = np.clip(0.08 + 0.45 * drought_index
                                + 0.20 * (ecosystem == "forest") + 0.14 * (ecozone == "Boreal Plains")
                                + 0.10 * (ecozone == "Montane Cordillera")
                                - 0.10 * wetness_index + rng.normal(0, 0.08, cfg.n_regions), 0, 1)

        pest_risk = np.clip(0.08 + 0.20 * temp_anomaly + 0.22 * (ecosystem == "forest")
                            + 0.12 * drought_index + rng.normal(0, 0.08, cfg.n_regions), 0, 1)

        # True conversion probability includes nonlinear interactions.
        logit_conversion = (
            -3.00
            + 1.45 * agriculture_pressure
            + 1.15 * urban_pressure
            + 0.75 * resource_pressure
            + 0.75 * degradation
            - 1.10 * protection_status
            + 0.55 * drought_index
            + 0.45 * (ecosystem == "grassland").astype(float)
            + 0.35 * (ecosystem == "wetland").astype(float)
            + 0.25 * (ecosystem == "peatland").astype(float)
            + 0.90 * agriculture_pressure * (ecosystem == "grassland").astype(float)
            + 0.65 * resource_pressure * (ecosystem == "peatland").astype(float)
        )
        conversion_probability = 1 / (1 + np.exp(-logit_conversion))

        # Observed conversion is stochastic.
        converted = rng.binomial(1, p=np.clip(conversion_probability, 0.001, 0.85))

        # True baseline emissions if no action, Mg CO2e/year.
        # This combines conversion, degradation, methane, and disturbance.
        methane_factor = (
            0.0
            + 10.0 * (ecosystem == "wetland").astype(float) * wetness_index
            + 18.0 * (ecosystem == "peatland").astype(float) * wetness_index * (1 - hydrology_integrity)
        )

        disturbance_emissions = carbon_density * area_km2 * 0.012 * (0.65 * wildfire_risk + 0.35 * pest_risk)

        true_baseline_ghg = (
            converted * carbon_density * area_km2 * rng.uniform(0.06, 0.16, cfg.n_regions)
            + degradation * carbon_density * area_km2 * 0.015
            + methane_factor * area_km2
            + disturbance_emissions
            + rng.normal(0, 6.0, cfg.n_regions)
        )
        true_baseline_ghg = np.maximum(true_baseline_ghg, 0)

        # Mechanistic approximation intentionally misses some nonlinearities.
        mech_baseline_ghg = (
            conversion_probability * carbon_density * area_km2 * 0.095
            + degradation * carbon_density * area_km2 * 0.012
            + methane_factor * area_km2 * 0.80
            + carbon_density * area_km2 * 0.010 * wildfire_risk
        )
        mech_baseline_ghg = np.maximum(mech_baseline_ghg, 0)

        for i in range(cfg.n_regions):
            rows.append({
                "region_id": region_id[i],
                "year": year,
                "ecozone": ecozone[i],
                "ecosystem": ecosystem[i],
                "x": x[i],
                "y": y[i],
                "area_km2": area_km2[i],
                "protection_status": protection_status[i],
                "indigenous_stewardship_context": indigenous_stewardship_context[i],
                "carbon_density": carbon_density[i],
                "biomass_carbon_density": biomass_carbon[i],
                "soil_peat_carbon_density": soil_peat_carbon[i],
                "agriculture_pressure": agriculture_pressure[i],
                "urban_pressure": urban_pressure[i],
                "resource_pressure": resource_pressure[i],
                "degradation": degradation[i],
                "hydrology_integrity": hydrology_integrity[i],
                "restoration_readiness": restoration_readiness[i],
                "monitoring_quality": monitoring_quality[i],
                "sar_overlap": sar_overlap[i],
                "connectivity": connectivity[i],
                "flood_exposure": flood_exposure[i],
                "heat_exposure": heat_exposure[i],
                "community_vulnerability": community_vulnerability[i],
                "temp_anomaly": temp_anomaly[i],
                "drought_index": drought_index[i],
                "wetness_index": wetness_index[i],
                "wildfire_risk": wildfire_risk[i],
                "pest_risk": pest_risk[i],
                "conversion_probability_true": conversion_probability[i],
                "converted_observed": converted[i],
                "true_baseline_ghg": true_baseline_ghg[i],
                "mechanistic_baseline_ghg": mech_baseline_ghg[i],
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# 2. Natural climate solution action scenarios
# ---------------------------------------------------------------------

def compute_action_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate net GHG benefit, biodiversity co-benefit, human well-being benefit,
    permanence risk, leakage/additionality risk, and uncertainty for three actions:
    protection, restoration, and improved management.
    """

    action_rows = []

    for action in ["protect", "restore", "manage"]:
        d = df.copy()
        d["action"] = action

        # Avoided emissions component.
        if action == "protect":
            additionality = np.clip(
                d["conversion_probability_true"] * (1 - d["protection_status"]) *
                (0.50 + 0.50 * d["carbon_density"] / d["carbon_density"].quantile(0.95)),
                0, 1
            )
            avoided = additionality * d["carbon_density"] * d["area_km2"] * 0.11

            carbon_gain = 0.03 * d["carbon_density"] * d["area_km2"] * d["connectivity"]
            methane_penalty = 0.0

        elif action == "restore":
            additionality = np.clip(d["restoration_readiness"] * d["degradation"], 0, 1)
            avoided = 0.25 * d["conversion_probability_true"] * d["carbon_density"] * d["area_km2"] * 0.07

            # Restoration recovers soil and biomass carbon, strongest when degraded.
            carbon_gain = (
                d["carbon_density"] * d["area_km2"]
                * (0.025 + 0.085 * d["degradation"])
                * d["restoration_readiness"]
            )

            # Methane penalty for rewetted wetlands/peatlands under high wetness.
            methane_penalty = (
                d["area_km2"] * d["wetness_index"]
                * (6.0 * (d["ecosystem"] == "wetland").astype(float)
                   + 11.0 * (d["ecosystem"] == "peatland").astype(float))
                * (1 - d["hydrology_integrity"])
            )

        else:  # manage
            additionality = np.clip(0.35 + 0.35 * d["degradation"] + 0.15 * d["monitoring_quality"], 0, 1)
            avoided = (
                d["carbon_density"] * d["area_km2"]
                * (0.035 * d["wildfire_risk"] + 0.020 * d["pest_risk"] + 0.018 * d["degradation"])
            )

            carbon_gain = (
                d["carbon_density"] * d["area_km2"]
                * (0.025 + 0.030 * d["connectivity"])
                * (d["ecosystem"].isin(["forest", "grassland", "riparian"]).astype(float) * 0.9 + 0.4)
            )
            methane_penalty = 0.5 * d["area_km2"] * d["wetness_index"] * (d["ecosystem"] == "wetland").astype(float)

        # Risk deductions.
        reversal_risk = np.clip(
            0.15 + 0.45 * d["wildfire_risk"] + 0.18 * d["drought_index"]
            + 0.10 * d["pest_risk"] - 0.12 * d["monitoring_quality"],
            0, 1
        )

        leakage_risk = np.clip(
            0.05 + 0.25 * d["agriculture_pressure"] + 0.20 * d["resource_pressure"]
            - 0.10 * d["indigenous_stewardship_context"],
            0, 1
        )

        implementation_disturbance = (
            0.006 * d["carbon_density"] * d["area_km2"] * (action == "restore")
            + 0.003 * d["carbon_density"] * d["area_km2"] * (action == "manage")
        )

        true_net_ghg_benefit = (
            avoided + carbon_gain - methane_penalty
            - reversal_risk * (0.18 * carbon_gain + 0.10 * avoided)
            - leakage_risk * 0.10 * avoided
            - implementation_disturbance
            + rng.normal(0, 8.0, len(d))
        )
        true_net_ghg_benefit = np.maximum(true_net_ghg_benefit, -25)

        # Mechanistic action model with simplified coefficients.
        mech_net_ghg_benefit = (
            0.90 * avoided + 0.82 * carbon_gain - 0.75 * methane_penalty
            - 0.13 * reversal_risk * (carbon_gain + avoided)
            - 0.08 * leakage_risk * avoided
            - implementation_disturbance
        )

        # Co-benefits.
        biodiversity_benefit = np.clip(
            0.22
            + 0.25 * d["sar_overlap"]
            + 0.23 * d["connectivity"]
            + 0.16 * d["restoration_readiness"] * (action == "restore")
            + 0.12 * (d["ecosystem"].isin(["wetland", "grassland", "riparian"]).astype(float))
            + 0.10 * d["indigenous_stewardship_context"]
            - 0.08 * d["urban_pressure"]
            + rng.normal(0, 0.04, len(d)),
            0, 1
        )

        human_wellbeing_benefit = np.clip(
            0.18
            + 0.27 * d["flood_exposure"] * (d["ecosystem"].isin(["wetland", "riparian"]).astype(float))
            + 0.20 * d["heat_exposure"] * (d["ecosystem"].isin(["forest", "riparian"]).astype(float))
            + 0.18 * d["community_vulnerability"]
            + 0.10 * d["indigenous_stewardship_context"]
            + 0.08 * d["monitoring_quality"]
            + rng.normal(0, 0.04, len(d)),
            0, 1
        )

        monitoring_readiness = np.clip(
            0.35 + 0.45 * d["monitoring_quality"] + 0.18 * d["protection_status"]
            + 0.12 * d["indigenous_stewardship_context"] - 0.10 * d["resource_pressure"],
            0, 1
        )

        # Higher uncertainty in data-poor, high-disturbance, high-methane systems.
        uncertainty = np.clip(
            0.15 + 0.35 * (1 - d["monitoring_quality"]) + 0.20 * reversal_risk
            + 0.10 * d["wetness_index"] * d["ecosystem"].isin(["wetland", "peatland"]).astype(float)
            + 0.08 * np.abs(true_net_ghg_benefit - mech_net_ghg_benefit) /
              (np.nanpercentile(np.abs(true_net_ghg_benefit - mech_net_ghg_benefit), 95) + 1e-9),
            0, 1
        )

        d["additionality"] = additionality
        d["avoided_emissions"] = avoided
        d["carbon_stock_gain"] = carbon_gain
        d["methane_penalty"] = methane_penalty
        d["reversal_risk"] = reversal_risk
        d["leakage_risk"] = leakage_risk
        d["implementation_disturbance"] = implementation_disturbance
        d["true_net_ghg_benefit"] = true_net_ghg_benefit
        d["mechanistic_net_ghg_benefit"] = mech_net_ghg_benefit
        d["biodiversity_benefit"] = biodiversity_benefit
        d["human_wellbeing_benefit"] = human_wellbeing_benefit
        d["monitoring_readiness"] = monitoring_readiness
        d["uncertainty_score"] = uncertainty

        action_rows.append(d)

    return pd.concat(action_rows, ignore_index=True)


# ---------------------------------------------------------------------
# 3. Mechanistic-only, AI-only, and hybrid model comparison
# ---------------------------------------------------------------------

def make_model_features(df: pd.DataFrame):
    categorical = ["ecozone", "ecosystem", "action"]
    numeric = [
        "area_km2", "protection_status", "indigenous_stewardship_context",
        "carbon_density", "biomass_carbon_density", "soil_peat_carbon_density",
        "agriculture_pressure", "urban_pressure", "resource_pressure",
        "degradation", "hydrology_integrity", "restoration_readiness",
        "monitoring_quality", "sar_overlap", "connectivity",
        "flood_exposure", "heat_exposure", "community_vulnerability",
        "temp_anomaly", "drought_index", "wetness_index",
        "wildfire_risk", "pest_risk", "conversion_probability_true",
        "mechanistic_net_ghg_benefit", "additionality",
        "avoided_emissions", "carbon_stock_gain", "methane_penalty",
        "reversal_risk", "leakage_risk"
    ]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop"
    )
    return pre, categorical + numeric


def evaluate_models(df: pd.DataFrame):
    """
    Use blocked validation by ecozone. This tests whether the hybrid approach
    generalizes to regions with different ecological structure.
    """

    y = df["true_net_ghg_benefit"].values
    mech_pred = df["mechanistic_net_ghg_benefit"].values
    residual_y = y - mech_pred

    pre, feature_names = make_model_features(df)

    ai_model = Pipeline([
        ("preprocess", pre),
        ("model", RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=5,
            random_state=CFG.random_seed,
            n_jobs=-1
        ))
    ])

    residual_model = Pipeline([
        ("preprocess", pre),
        ("model", GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.035,
            max_depth=3,
            random_state=CFG.random_seed
        ))
    ])

    groups = df["ecozone"].values
    gkf = GroupKFold(n_splits=5)

    preds = np.zeros((len(df), 3))
    preds[:, 0] = mech_pred

    fold_records = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(df, y, groups=groups), start=1):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        ai_model.fit(train, y[train_idx])
        pred_ai = ai_model.predict(test)

        residual_model.fit(train, residual_y[train_idx])
        pred_resid = residual_model.predict(test)
        pred_hybrid = test["mechanistic_net_ghg_benefit"].values + pred_resid

        preds[test_idx, 1] = pred_ai
        preds[test_idx, 2] = pred_hybrid

        for model_name, pred in [
            ("Mechanistic only", test["mechanistic_net_ghg_benefit"].values),
            ("AI only", pred_ai),
            ("Hybrid mechanistic + AI residual", pred_hybrid),
        ]:
            fold_records.append({
                "fold": fold,
                "heldout_ecozones": ", ".join(sorted(test["ecozone"].unique())),
                "model": model_name,
                "R2": r2_score(y[test_idx], pred),
                "RMSE": rmse(y[test_idx], pred),
                "MAE": mean_absolute_error(y[test_idx], pred),
                "Bias": float(np.mean(pred - y[test_idx])),
            })

    performance = pd.DataFrame(fold_records)
    overall = []
    for j, model_name in enumerate(["Mechanistic only", "AI only", "Hybrid mechanistic + AI residual"]):
        pred = preds[:, j]
        overall.append({
            "model": model_name,
            "R2": r2_score(y, pred),
            "RMSE": rmse(y, pred),
            "MAE": mean_absolute_error(y, pred),
            "Bias": float(np.mean(pred - y)),
        })
    overall = pd.DataFrame(overall)

    df_pred = df.copy()
    df_pred["pred_mechanistic"] = preds[:, 0]
    df_pred["pred_ai_only"] = preds[:, 1]
    df_pred["pred_hybrid"] = preds[:, 2]
    df_pred["hybrid_residual_correction"] = df_pred["pred_hybrid"] - df_pred["pred_mechanistic"]

    return df_pred, performance, overall, residual_model


# ---------------------------------------------------------------------
# 4. Uncertainty via bootstrap residual learners
# ---------------------------------------------------------------------

def bootstrap_uncertainty(df: pd.DataFrame, n_bootstrap: int = 40) -> pd.DataFrame:
    """
    Approximate predictive uncertainty using bootstrap residual models.
    This gives a practical proof-of-concept for uncertainty bands.
    """

    y = df["true_net_ghg_benefit"].values
    mech = df["mechanistic_net_ghg_benefit"].values
    residual_y = y - mech

    pre, _ = make_model_features(df)
    preds = []

    n = len(df)
    for b in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        model = Pipeline([
            ("preprocess", pre),
            ("model", GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.04,
                max_depth=3,
                random_state=CFG.random_seed + b
            ))
        ])
        model.fit(df.iloc[idx], residual_y[idx])
        preds.append(mech + model.predict(df))

    arr = np.vstack(preds)
    out = df.copy()
    out["pred_hybrid_boot_mean"] = arr.mean(axis=0)
    out["pred_hybrid_p05"] = np.percentile(arr, 5, axis=0)
    out["pred_hybrid_p95"] = np.percentile(arr, 95, axis=0)
    out["pred_hybrid_interval_width"] = out["pred_hybrid_p95"] - out["pred_hybrid_p05"]
    out["interval_contains_truth"] = (
        (out["true_net_ghg_benefit"] >= out["pred_hybrid_p05"])
        & (out["true_net_ghg_benefit"] <= out["pred_hybrid_p95"])
    )
    return out


# ---------------------------------------------------------------------
# 5. Decision score, priority ranking, sensitivity, identifiability
# ---------------------------------------------------------------------

def compute_priority_score(df: pd.DataFrame, weights=None) -> pd.DataFrame:
    if weights is None:
        weights = {
            "ghg": 0.35,
            "biodiversity": 0.22,
            "wellbeing": 0.16,
            "readiness": 0.12,
            "uncertainty": 0.08,
            "permanence": 0.07,
        }

    d = df.copy()
    d["ghg_scaled"] = safe_scale(d["pred_hybrid"])
    d["biodiversity_scaled"] = safe_scale(d["biodiversity_benefit"])
    d["wellbeing_scaled"] = safe_scale(d["human_wellbeing_benefit"])
    d["readiness_scaled"] = safe_scale(d["monitoring_readiness"])
    d["uncertainty_scaled"] = safe_scale(d["uncertainty_score"])
    d["permanence_risk_scaled"] = safe_scale(d["reversal_risk"])

    d["priority_score"] = (
        weights["ghg"] * d["ghg_scaled"]
        + weights["biodiversity"] * d["biodiversity_scaled"]
        + weights["wellbeing"] * d["wellbeing_scaled"]
        + weights["readiness"] * d["readiness_scaled"]
        - weights["uncertainty"] * d["uncertainty_scaled"]
        - weights["permanence"] * d["permanence_risk_scaled"]
    )

    return d


def decision_weight_sensitivity(df: pd.DataFrame, n_samples: int = 300) -> pd.DataFrame:
    """
    Vary decision weights using a Dirichlet distribution and measure how often
    each region-action appears in the top 5 percent.
    """

    key_cols = ["region_id", "year", "ecozone", "ecosystem", "action"]
    base = df[key_cols].copy()
    base["top_count"] = 0

    for _ in range(n_samples):
        w_pos = rng.dirichlet([4.0, 2.5, 2.0, 1.4])
        w_neg = rng.dirichlet([1.3, 1.3]) * 0.20
        weights = {
            "ghg": float(w_pos[0] * 0.80),
            "biodiversity": float(w_pos[1] * 0.80),
            "wellbeing": float(w_pos[2] * 0.80),
            "readiness": float(w_pos[3] * 0.80),
            "uncertainty": float(w_neg[0]),
            "permanence": float(w_neg[1]),
        }

        scored = compute_priority_score(df, weights)
        threshold = scored["priority_score"].quantile(0.95)
        base["top_count"] += (scored["priority_score"] >= threshold).astype(int).values

    base["top5_frequency"] = base["top_count"] / n_samples
    return base


def parameter_sensitivity(df: pd.DataFrame, n_samples: int = 300) -> pd.DataFrame:
    """
    Test sensitivity to methane penalty, reversal deduction, leakage deduction,
    and carbon-gain multiplier.
    """

    records = []
    sample_df = df.sample(n=min(5000, len(df)), random_state=CFG.random_seed).copy()

    for i in range(n_samples):
        methane_mult = rng.uniform(0.5, 1.8)
        reversal_mult = rng.uniform(0.5, 1.8)
        leakage_mult = rng.uniform(0.4, 1.6)
        carbon_gain_mult = rng.uniform(0.7, 1.4)

        perturbed_ghg = (
            sample_df["avoided_emissions"]
            + carbon_gain_mult * sample_df["carbon_stock_gain"]
            - methane_mult * sample_df["methane_penalty"]
            - reversal_mult * sample_df["reversal_risk"] *
              (0.18 * carbon_gain_mult * sample_df["carbon_stock_gain"] + 0.10 * sample_df["avoided_emissions"])
            - leakage_mult * sample_df["leakage_risk"] * 0.10 * sample_df["avoided_emissions"]
            - sample_df["implementation_disturbance"]
        )

        records.append({
            "iteration": i,
            "methane_multiplier": methane_mult,
            "reversal_multiplier": reversal_mult,
            "leakage_multiplier": leakage_mult,
            "carbon_gain_multiplier": carbon_gain_mult,
            "mean_net_ghg_benefit": float(np.mean(perturbed_ghg)),
            "p10_net_ghg_benefit": float(np.percentile(perturbed_ghg, 10)),
            "p90_net_ghg_benefit": float(np.percentile(perturbed_ghg, 90)),
        })

    return pd.DataFrame(records)


def parameter_bootstrap_identifiability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a simplified linear accounting equation repeatedly to quantify
    parameter stability and correlation. This demonstrates identifiability:
    if two mechanisms are hard to separate, their bootstrap estimates become
    strongly correlated or unstable.
    """

    cols = [
        "avoided_emissions",
        "carbon_stock_gain",
        "methane_penalty",
        "reversal_risk",
        "leakage_risk",
        "implementation_disturbance",
    ]

    d = df.sample(n=min(9000, len(df)), random_state=CFG.random_seed).copy()
    X = d[cols].copy()
    X["reversal_exposure"] = d["reversal_risk"] * (d["avoided_emissions"] + d["carbon_stock_gain"])
    X["leakage_exposure"] = d["leakage_risk"] * d["avoided_emissions"]
    X = X[[
        "avoided_emissions",
        "carbon_stock_gain",
        "methane_penalty",
        "reversal_exposure",
        "leakage_exposure",
        "implementation_disturbance",
    ]]
    y = d["true_net_ghg_benefit"].values

    records = []
    for b in range(CFG.n_bootstrap):
        idx = rng.choice(np.arange(len(d)), size=len(d), replace=True)
        lr = LinearRegression()
        lr.fit(X.iloc[idx], y[idx])
        rec = {"bootstrap": b}
        for name, coef in zip(X.columns, lr.coef_):
            rec[name] = coef
        rec["intercept"] = lr.intercept_
        rec["R2"] = lr.score(X.iloc[idx], y[idx])
        records.append(rec)

    coef_df = pd.DataFrame(records)
    summary = []
    for col in X.columns:
        summary.append({
            "parameter": col,
            "mean_estimate": coef_df[col].mean(),
            "sd_estimate": coef_df[col].std(),
            "p05": coef_df[col].quantile(0.05),
            "p95": coef_df[col].quantile(0.95),
            "coefficient_of_variation_abs": abs(coef_df[col].std() / (coef_df[col].mean() + 1e-9)),
        })
    summary = pd.DataFrame(summary)

    corr = coef_df[X.columns].corr()
    corr.to_csv(OUT / "parameter_bootstrap_correlation.csv", index=True)
    coef_df.to_csv(OUT / "parameter_bootstrap_coefficients.csv", index=False)
    return summary


# ---------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------

def plot_outputs(df: pd.DataFrame, performance: pd.DataFrame, overall: pd.DataFrame,
                 weight_sens: pd.DataFrame, param_sens: pd.DataFrame,
                 param_summary: pd.DataFrame):
    # Model performance bar plot.
    plt.figure(figsize=(8, 4.8))
    order = ["Mechanistic only", "AI only", "Hybrid mechanistic + AI residual"]
    perf = overall.set_index("model").loc[order].reset_index()
    plt.bar(perf["model"], perf["RMSE"])
    plt.ylabel("RMSE: net GHG benefit")
    plt.title("Blocked-validation model error")
    plt.xticks(rotation=20, ha="right")
    savefig("fig01_model_rmse.png")

    # Predicted vs true for hybrid.
    plt.figure(figsize=(6, 5.5))
    sample = df.sample(n=min(5000, len(df)), random_state=CFG.random_seed)
    plt.scatter(sample["true_net_ghg_benefit"], sample["pred_hybrid"], s=8, alpha=0.35)
    lim = [
        min(sample["true_net_ghg_benefit"].min(), sample["pred_hybrid"].min()),
        max(sample["true_net_ghg_benefit"].max(), sample["pred_hybrid"].max())
    ]
    plt.plot(lim, lim, linestyle="--")
    plt.xlabel("True simulated net GHG benefit")
    plt.ylabel("Hybrid predicted net GHG benefit")
    plt.title("Hybrid mechanistic-AI prediction")
    savefig("fig02_hybrid_predicted_vs_true.png")

    # Scenario mean outcomes.
    scenario_summary = (
        df.groupby(["ecosystem", "action"], as_index=False)
        .agg(mean_ghg=("pred_hybrid", "mean"),
             mean_biodiversity=("biodiversity_benefit", "mean"),
             mean_wellbeing=("human_wellbeing_benefit", "mean"),
             mean_uncertainty=("uncertainty_score", "mean"))
    )
    scenario_summary.to_csv(OUT / "ecosystem_action_summary.csv", index=False)

    plt.figure(figsize=(9, 5))
    pivot = scenario_summary.pivot(index="ecosystem", columns="action", values="mean_ghg")
    pivot.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean predicted net GHG benefit")
    plt.title("Scenario outcomes by ecosystem")
    plt.xticks(rotation=20, ha="right")
    savefig("fig03_scenario_outcomes_by_ecosystem.png")

    # Carbon-biodiversity-wellbeing trade-off.
    plt.figure(figsize=(6.2, 5.4))
    sample = df.sample(n=min(6000, len(df)), random_state=CFG.random_seed + 2)
    plt.scatter(sample["pred_hybrid"], sample["biodiversity_benefit"],
                s=10, alpha=0.35)
    plt.xlabel("Predicted net GHG benefit")
    plt.ylabel("Biodiversity co-benefit")
    plt.title("Carbon-biodiversity trade-off and synergy space")
    savefig("fig04_carbon_biodiversity_synergy.png")

    # Priority map.
    latest = df[df["year"] == df["year"].max()].copy()
    latest = latest.sort_values("priority_score", ascending=False)
    threshold = latest["priority_score"].quantile(0.95)
    latest["top_priority"] = latest["priority_score"] >= threshold

    plt.figure(figsize=(7, 6))
    plt.scatter(latest["x"], latest["y"], s=12, alpha=0.35)
    top = latest[latest["top_priority"]]
    plt.scatter(top["x"], top["y"], s=30, marker="^")
    plt.xlabel("Synthetic west-east coordinate")
    plt.ylabel("Synthetic south-north coordinate")
    plt.title("Synthetic priority map: top 5 percent highlighted")
    savefig("fig05_priority_map.png")

    # Uncertainty interval width by ecosystem.
    plt.figure(figsize=(8, 4.8))
    df.boxplot(column="pred_hybrid_interval_width", by="ecosystem", grid=False)
    plt.suptitle("")
    plt.title("Hybrid predictive uncertainty by ecosystem")
    plt.xlabel("Ecosystem")
    plt.ylabel("90 percent interval width")
    plt.xticks(rotation=20, ha="right")
    savefig("fig06_uncertainty_by_ecosystem.png")

    # Sensitivity of top-priority stability.
    merged = df[["region_id", "year", "ecozone", "ecosystem", "action", "priority_score"]].merge(
        weight_sens,
        on=["region_id", "year", "ecozone", "ecosystem", "action"],
        how="left"
    )
    plt.figure(figsize=(6.2, 5.2))
    sample = merged.sample(n=min(6000, len(merged)), random_state=CFG.random_seed)
    plt.scatter(sample["priority_score"], sample["top5_frequency"], s=9, alpha=0.35)
    plt.xlabel("Base priority score")
    plt.ylabel("Frequency in top 5 percent under weight changes")
    plt.title("Decision-weight sensitivity")
    savefig("fig07_weight_sensitivity.png")

    # Parameter sensitivity tornado-like correlation.
    drivers = ["methane_multiplier", "reversal_multiplier", "leakage_multiplier", "carbon_gain_multiplier"]
    corrs = []
    for col in drivers:
        corrs.append({
            "parameter": col,
            "correlation_with_mean_outcome": param_sens[col].corr(param_sens["mean_net_ghg_benefit"])
        })
    corr_df = pd.DataFrame(corrs)
    corr_df.to_csv(OUT / "parameter_sensitivity_correlations.csv", index=False)

    plt.figure(figsize=(7.2, 4.6))
    plt.barh(corr_df["parameter"], corr_df["correlation_with_mean_outcome"])
    plt.xlabel("Correlation with mean net GHG benefit")
    plt.title("Sensitivity of climate-benefit estimate to accounting assumptions")
    savefig("fig08_parameter_sensitivity.png")

    # Bootstrap identifiability.
    plt.figure(figsize=(8, 4.8))
    plt.bar(param_summary["parameter"], param_summary["coefficient_of_variation_abs"])
    plt.ylabel("Absolute coefficient of variation")
    plt.title("Parameter identifiability from bootstrap accounting fits")
    plt.xticks(rotation=25, ha="right")
    savefig("fig09_parameter_identifiability.png")


# ---------------------------------------------------------------------
# 7. Main workflow
# ---------------------------------------------------------------------

def main():
    print_header("NATURE-CARBON Canada proof-of-concept simulation")

    print("1. Simulating harmonized nature-carbon landscape database...")
    landscape = simulate_landscape_database(CFG)
    landscape.to_csv(OUT / "synthetic_landscape_database.csv", index=False)
    print(f"   Saved {len(landscape):,} region-year records.")

    print("2. Simulating natural climate solution action scenarios...")
    scenarios = compute_action_outcomes(landscape)
    scenarios.to_csv(OUT / "scenario_results_raw.csv", index=False)
    print(f"   Saved {len(scenarios):,} region-year-action records.")

    print("3. Evaluating mechanistic-only, AI-only, and hybrid models...")
    pred_df, fold_perf, overall_perf, residual_model = evaluate_models(scenarios)
    fold_perf.to_csv(OUT / "model_performance_by_fold.csv", index=False)
    overall_perf.to_csv(OUT / "model_performance.csv", index=False)
    print(overall_perf.to_string(index=False))

    print("4. Estimating bootstrap predictive uncertainty...")
    uncertain_df = bootstrap_uncertainty(pred_df, n_bootstrap=CFG.n_bootstrap)
    coverage = uncertain_df["interval_contains_truth"].mean()
    print(f"   Empirical 90% interval coverage: {coverage:.3f}")

    print("5. Computing transparent decision-priority scores...")
    scored = compute_priority_score(uncertain_df)
    scored.to_csv(OUT / "scenario_results.csv", index=False)

    top = (
        scored.sort_values("priority_score", ascending=False)
        .head(50)[[
            "region_id", "year", "ecozone", "ecosystem", "action",
            "pred_hybrid", "biodiversity_benefit", "human_wellbeing_benefit",
            "reversal_risk", "leakage_risk", "uncertainty_score",
            "monitoring_readiness", "priority_score"
        ]]
    )
    top.to_csv(OUT / "top_priority_regions.csv", index=False)

    print("6. Running decision-weight sensitivity analysis...")
    weight_sens = decision_weight_sensitivity(scored, n_samples=CFG.n_weight_samples)
    weight_sens.to_csv(OUT / "sensitivity_weights.csv", index=False)

    print("7. Running accounting-parameter sensitivity analysis...")
    param_sens = parameter_sensitivity(scored, n_samples=CFG.n_parameter_samples)
    param_sens.to_csv(OUT / "sensitivity_parameters.csv", index=False)

    print("8. Running bootstrap parameter-identifiability analysis...")
    param_summary = parameter_bootstrap_identifiability(scored)
    param_summary.to_csv(OUT / "parameter_bootstrap_summary.csv", index=False)
    print(param_summary.to_string(index=False))

    print("9. Creating figures...")
    plot_outputs(scored, fold_perf, overall_perf, weight_sens, param_sens, param_summary)

    print_header("Completed")
    print(f"All outputs saved to: {OUT.resolve()}")
    print("\nKey result to report:")
    best = overall_perf.sort_values("RMSE").iloc[0]
    print(f"Best model: {best['model']} | R2={best['R2']:.3f}, RMSE={best['RMSE']:.2f}, MAE={best['MAE']:.2f}")
    print(f"Top-priority table: {OUT / 'top_priority_regions.csv'}")
    print(f"Main scenario table: {OUT / 'scenario_results.csv'}")


if __name__ == "__main__":
    main()
