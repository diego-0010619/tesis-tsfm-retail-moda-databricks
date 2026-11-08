"""
Paquete modular `fashion_tsfm` para el Proyecto de Grado de Maestría en Ciencia de Datos (UPB).

Título: Modelos fundacionales de series de tiempo a escala Big Data para el
        pronóstico de demanda en el retail de moda.
Autor:  Diego Alexander Sánchez Henao
Director: Ferney Orlando Amaya Fernández
Grupo:  GIDATIC - Universidad Pontificia Bolivariana (Sede Medellín)
"""

from .config import (
    ExperimentConfig,
    ensure_unity_catalog_infrastructure,
    get_project_root,
    load_experiment_config,
)
from .cost_profiler import (
    CostProfiler,
    get_compute_environment_info,
    profile_forecasting_step,
)
from .metrics_and_inventory import (
    compute_forecast_metrics_pandas,
    compute_forecast_metrics_spark,
    compute_service_efficiency_curve,
    simulate_periodic_review_inventory,
)
from .regime_classifier import (
    classify_demand_regimes_pandas,
    classify_demand_regimes_spark,
    summarize_regimes_pandas,
    summarize_regimes_spark,
)

__version__ = "1.0.0"

__all__ = [
    "ExperimentConfig",
    "load_experiment_config",
    "get_project_root",
    "ensure_unity_catalog_infrastructure",
    "classify_demand_regimes_spark",
    "classify_demand_regimes_pandas",
    "summarize_regimes_spark",
    "summarize_regimes_pandas",
    "compute_forecast_metrics_pandas",
    "compute_forecast_metrics_spark",
    "simulate_periodic_review_inventory",
    "compute_service_efficiency_curve",
    "CostProfiler",
    "profile_forecasting_step",
    "get_compute_environment_info",
]

