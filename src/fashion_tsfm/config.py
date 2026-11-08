"""
Módulo de gestión de configuración YAML e inicialización de Unity Catalog en Databricks.

Resuelve automáticamente rutas locales o dentro de Databricks Git Folders (Repos),
valida el catálogo disponible en Databricks (ej. `workspace`, `main` o `hive_metastore`),
y provee métodos limpios para obtener los nombres calificados (FQN) de las tablas
en las capas Bronze, Silver y Gold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def _default_config_dict() -> Dict[str, Any]:
    """
    Diccionario de respaldo idéntico a `configs/experiment_config.yaml` en caso de que
    el intérprete aún no tenga instalado `PyYAML`.
    """
    return {
        "project": {
            "name": "tesis-tsfm-retail-moda-databricks",
            "title": "Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda",
            "author": "Diego Alexander Sánchez Henao",
            "advisor": "Ferney Orlando Amaya Fernández",
            "institution": "Universidad Pontificia Bolivariana (UPB) - Sede Medellín",
            "research_group": "GIDATIC (Categoría A - Minciencias)",
            "random_seed": 42,
            "mlflow_experiment_name": "/Shared/tesis_tsfm_retail_moda_upb",
        },
        "unity_catalog": {
            "catalog": "workspace",
            "fallback_catalogs": ["main", "hive_metastore"],
            "schema": "fashion_forecasting",
            "volume_name": "raw_data",
            "volume_path": "/Volumes/workspace/fashion_forecasting/raw_data",
            "tables": {
                "bronze_hm_transactions": "bronze_hm_transactions",
                "bronze_hm_articles": "bronze_hm_articles",
                "bronze_hm_customers": "bronze_hm_customers",
                "bronze_visuelle": "bronze_visuelle_coldstart",
                "bronze_m5_control": "bronze_m5_control",
                "bronze_local_retailer": "bronze_colombia_retailer",
                "silver_weekly_panel": "silver_fashion_weekly_panel",
                "silver_regime_catalog": "silver_regime_catalog",
                "gold_forecasts_baselines": "gold_forecasts_baselines",
                "gold_forecasts_tsfm": "gold_forecasts_tsfm",
                "gold_scaling_benchmarks": "gold_scaling_benchmarks",
                "gold_inventory_simulation": "gold_inventory_simulation",
                "gold_benchmark_summary": "gold_benchmark_summary",
                "gold_decision_matrix": "gold_decision_matrix",
            },
        },
        "execution": {
            "use_smoke_test": True,
            "active_dataset": "hm",
            "raw_files": {
                "hm_transactions": "hm/transactions_train.csv",
                "hm_articles": "hm/articles.csv",
                "hm_customers": "hm/customers.csv",
                "visuelle_sales": "visuelle/visuelle_weekly_sales.csv",
                "m5_sales": "m5/sales_train_evaluation.csv",
                "local_colombia_sales": "local_colombia/ventas_semanales_sku.csv",
            },
        },
        "smoke_test": {
            "n_articles": 500,
            "n_channels": 2,
            "n_weeks": 60,
            "start_date": "2019-07-01",
            "regime_proportions": {
                "Smooth": 0.25,
                "Erratic": 0.15,
                "Intermittent": 0.30,
                "Lumpy": 0.20,
                "Cold_Start": 0.10,
            },
        },
        "forecasting": {
            "frequency": "W-MON",
            "horizons": [4, 8, 12],
            "primary_horizon": 12,
            "context_lengths": [12, 26, 52],
            "primary_context_length": 26,
            "quantiles": [0.1, 0.5, 0.9],
            "seasonal_period": 52,
            "validation_weeks": 8,
            "test_weeks": 12,
            "exogenous_covariates": [
                "avg_price",
                "discount_depth",
                "sales_channel_id",
                "month_sin",
                "month_cos",
                "is_black_friday",
                "is_christmas",
                "is_mothers_day_co",
                "is_amor_y_amistad_co",
            ],
        },
        "regime_thresholds": {
            "adi_cutoff": 1.32,
            "cv2_cutoff": 0.49,
            "cold_start_max_weeks": 6,
        },
        "models": {
            "baselines": {
                "statistical_models": ["CrostonClassic", "CrostonSBA", "TSB", "AutoETS", "SeasonalNaive"],
                "lightgbm": {"n_estimators": 250, "learning_rate": 0.05, "num_leaves": 31},
            },
            "tsfm": {
                "chronos_bolt": {"repo_id": "amazon/chronos-bolt-small", "fallback_repo_id": "amazon/chronos-bolt-tiny"},
                "covariate_adapter_ridge_alpha": 1.0,
            },
            "lora_finetuning": {
                "base_model_id": "amazon/chronos-t5-small",
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "learning_rate": 0.0005,
                "batch_size": 32,
                "epochs": 3,
            },
        },
        "scaling_benchmark": {
            "sample_sizes": [100, 1000, 10000, 100000],
            "default_spark_partitions": 16,
            "batch_size_cpu": 32,
            "batch_size_gpu": 128,
        },
        "inventory_simulation": {
            "lead_time_weeks": 2,
            "review_period_weeks": 1,
            "target_service_levels": [0.80, 0.90, 0.95],
            "primary_service_level": 0.90,
            "holding_cost_per_unit_week": 0.50,
            "stockout_penalty_per_unit": 5.00,
            "initial_stock_weeks_cover": 2.0,
        },
        "governance": {
            "hash_salt_env_var": "FASHION_HASH_SALT",
            "default_salt": "upb_gidatic_ley1581_2026_salt",
        },
    }


def get_project_root() -> Path:
    """
    Detecta la raíz del repositorio tanto en ejecución local como dentro de
    Databricks Git Folders (/Workspace/Users/.../tesis-tsfm-retail-moda-databricks).
    """
    try:
        current_file = Path(__file__).resolve()
        # src/fashion_tsfm/config.py -> subimos 2 niveles para llegar a la raíz
        candidate_root = current_file.parents[2]
        if (candidate_root / "configs" / "experiment_config.yaml").exists():
            return candidate_root
    except NameError:
        pass

    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent, cwd.parent.parent]:
        if (candidate / "configs" / "experiment_config.yaml").exists():
            return candidate
    return cwd


@dataclass
class ExperimentConfig:
    """
    Contenedor estructurado para acceder a los parámetros de `configs/experiment_config.yaml`
    y resolver nombres de tablas y rutas de volúmenes en Unity Catalog.
    """

    raw: Dict[str, Any] = field(default_factory=dict)
    active_catalog: str = "workspace"

    def __post_init__(self) -> None:
        uc = self.raw.get("unity_catalog", {})
        self.active_catalog = uc.get("catalog", "workspace")

    @property
    def project_meta(self) -> Dict[str, Any]:
        return self.raw.get("project", {})

    @property
    def schema_name(self) -> str:
        return self.raw.get("unity_catalog", {}).get("schema", "fashion_forecasting")

    @property
    def volume_name(self) -> str:
        return self.raw.get("unity_catalog", {}).get("volume_name", "raw_data")

    @property
    def volume_path(self) -> str:
        """
        Retorna la ruta POSIX del Volumen en Unity Catalog:
        `/Volumes/<active_catalog>/<schema>/<volume_name>`
        """
        return f"/Volumes/{self.active_catalog}/{self.schema_name}/{self.volume_name}"

    @property
    def smoke_test_data_path(self) -> str:
        """Ruta dentro del volumen donde se guardan los archivos del Smoke Test."""
        return f"{self.volume_path}/smoke_test"

    @property
    def figures_output_path(self) -> str:
        """Ruta donde se exportan las figuras de alta resolución para la tesis."""
        return f"{self.volume_path}/thesis_figures"

    @property
    def use_smoke_test(self) -> bool:
        return bool(self.raw.get("execution", {}).get("use_smoke_test", True))

    @property
    def active_dataset(self) -> str:
        return str(self.raw.get("execution", {}).get("active_dataset", "hm"))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("project", {}).get("random_seed", 42))

    @property
    def primary_horizon(self) -> int:
        return int(self.raw.get("forecasting", {}).get("primary_horizon", 12))

    @property
    def horizons(self) -> List[int]:
        return list(self.raw.get("forecasting", {}).get("horizons", [4, 8, 12]))

    @property
    def primary_context_length(self) -> int:
        return int(self.raw.get("forecasting", {}).get("primary_context_length", 26))

    @property
    def context_lengths(self) -> List[int]:
        return list(self.raw.get("forecasting", {}).get("context_lengths", [12, 26, 52]))

    @property
    def quantiles(self) -> List[float]:
        return [float(q) for q in self.raw.get("forecasting", {}).get("quantiles", [0.1, 0.5, 0.9])]

    @property
    def validation_weeks(self) -> int:
        return int(self.raw.get("forecasting", {}).get("validation_weeks", 8))

    @property
    def test_weeks(self) -> int:
        return int(self.raw.get("forecasting", {}).get("test_weeks", 12))

    @property
    def exogenous_covariates(self) -> List[str]:
        return list(self.raw.get("forecasting", {}).get("exogenous_covariates", []))

    @property
    def regime_thresholds(self) -> Dict[str, float]:
        defaults = {"adi_cutoff": 1.32, "cv2_cutoff": 0.49, "cold_start_max_weeks": 6}
        defaults.update(self.raw.get("regime_thresholds", {}))
        return defaults

    @property
    def inventory_params(self) -> Dict[str, Any]:
        return self.raw.get("inventory_simulation", {})

    @property
    def scaling_params(self) -> Dict[str, Any]:
        return self.raw.get("scaling_benchmark", {})

    @property
    def governance_salt(self) -> str:
        gov = self.raw.get("governance", {})
        env_var = gov.get("hash_salt_env_var", "FASHION_HASH_SALT")
        default_salt = gov.get("default_salt", "upb_gidatic_ley1581_2026_salt")
        return os.environ.get(env_var, default_salt)

    def get_table_fqn(self, table_key: str) -> str:
        """
        Retorna el nombre completo en formato `catalog.schema.table_name` para Unity Catalog.
        Acepta ya sea la llave definida en `unity_catalog.tables` o el nombre directo.
        """
        tables_dict = self.raw.get("unity_catalog", {}).get("tables", {})
        table_name = tables_dict.get(table_key, table_key)
        return f"{self.active_catalog}.{self.schema_name}.{table_name}"

    def get_raw_file_path(self, file_key: str) -> str:
        """
        Retorna la ruta absoluta en el Volumen para un archivo de datos real
        definido en `execution.raw_files`.
        """
        raw_files = self.raw.get("execution", {}).get("raw_files", {})
        rel_path = raw_files.get(file_key, file_key)
        return f"{self.volume_path}/{rel_path}"


def load_experiment_config(config_path: Optional[str | Path] = None) -> ExperimentConfig:
    """
    Carga el archivo YAML de configuración experimental y devuelve una instancia
    de `ExperimentConfig`.
    """
    if config_path is None:
        root = get_project_root()
        resolved_path = root / "configs" / "experiment_config.yaml"
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración en: {resolved_path}. "
            "Verifica que estás ejecutando desde una carpeta dentro del repositorio."
        )

    if yaml is not None:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
    else:
        raw_dict = _default_config_dict()

    return ExperimentConfig(raw=raw_dict)


def ensure_unity_catalog_infrastructure(spark: Any, cfg: ExperimentConfig) -> Dict[str, str]:
    """
    Verifica qué catálogo de Unity Catalog está activo en el Workspace de Databricks
    (probando `cfg.active_catalog` y los `fallback_catalogs`), crea el esquema
    `fashion_forecasting` y el volumen `raw_data` si no existen, y actualiza
    `cfg.active_catalog` en caso de fallback.

    Parameters
    ----------
    spark : SparkSession
        Sesión activa de PySpark en Databricks.
    cfg : ExperimentConfig
        Configuración cargada del experimento.

    Returns
    -------
    Dict[str, str]
        Diccionario con el catálogo, esquema y ruta de volumen inicializados.
    """
    uc_cfg = cfg.raw.get("unity_catalog", {})
    preferred_catalog = uc_cfg.get("catalog", "workspace")
    fallbacks = uc_cfg.get("fallback_catalogs", ["main", "hive_metastore"])
    candidate_catalogs = [preferred_catalog] + [c for c in fallbacks if c != preferred_catalog]

    available_catalogs: List[str] = []
    try:
        catalog_rows = spark.sql("SHOW CATALOGS").collect()
        available_catalogs = [
            str(row[0]).lower() for row in catalog_rows if row and len(row) > 0
        ]
    except Exception:
        available_catalogs = []

    selected_catalog = preferred_catalog
    if available_catalogs:
        for cand in candidate_catalogs:
            if cand.lower() in available_catalogs:
                selected_catalog = cand
                break
        else:
            # Si ninguno de los candidatos coincide, tomamos el catálogo actual de la sesión
            try:
                current_cat = spark.sql("SELECT current_catalog()").collect()[0][0]
                if current_cat:
                    selected_catalog = str(current_cat)
            except Exception:
                selected_catalog = available_catalogs[0]

    cfg.active_catalog = selected_catalog
    schema_fqn = f"{cfg.active_catalog}.{cfg.schema_name}"
    volume_fqn = f"{cfg.active_catalog}.{cfg.schema_name}.{cfg.volume_name}"

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_fqn}")

    volume_status = "CREATED_OR_VERIFIED"
    if cfg.active_catalog.lower() != "hive_metastore":
        try:
            spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_fqn}")
        except Exception as exc:
            volume_status = f"FALLBACK_DBFS ({exc})"
    else:
        volume_status = "HIVE_METASTORE_NO_UC_VOLUME"

    # Crear subdirectorios auxiliares en el volumen si el sistema de archivos es accesible
    for subdir in [cfg.smoke_test_data_path, cfg.figures_output_path]:
        try:
            Path(subdir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    return {
        "active_catalog": cfg.active_catalog,
        "schema_fqn": schema_fqn,
        "volume_fqn": volume_fqn,
        "volume_path": cfg.volume_path,
        "volume_status": volume_status,
    }
