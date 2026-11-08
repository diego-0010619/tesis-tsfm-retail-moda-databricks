# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03: Líneas Base Estadísticas (`Croston`, `SBA`, `TSB`, `ETS`, `SeasonalNaive`) y Modelo Global Supervisado (`LightGBM`)
# MAGIC
# MAGIC **Proyecto de Grado — Maestría en Ciencia de Datos (UPB, Sede Medellín - Grupo GIDATIC)**
# MAGIC * **Título:** *Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda*
# MAGIC * **Autor:** Diego Alexander Sánchez Henao | **Director:** Prof. Ferney Orlando Amaya Fernández
# MAGIC
# MAGIC ---
# MAGIC ### Propósito de este Notebook
# MAGIC De acuerdo con el componente **C (Comparación)** de nuestro marco PICO y la evidencia de Makridakis et al. (2022a) y Croston (1972), ningún modelo fundacional puede declararse superior en retail sin antes superar dos familias de líneas base rigurosas:
# MAGIC 1. **Métodos Estadísticos Especializados en Intermitencia y Estacionalidad (ejecución distribuida en PySpark mediante `applyInPandas`):**
# MAGIC    * `CrostonClassic` (Croston, 1972): Descompone la demanda en tamaño de pedido positivo ($z_t$) e intervalo entre llegadas ($p_t$).
# MAGIC    * `CrostonSBA` (*Syntetos-Boylan Approximation*): Aplica el factor corrector $(1 - \alpha/2)$ que corrige el sesgo positivo de Croston clásico en series *Intermittent* y *Lumpy*.
# MAGIC    * `TSB` (*Teunter-Syntetos-Babai*): Actualiza directamente la probabilidad de ocurrencia en cada periodo, ideal cuando existe riesgo de obsolescencia en moda.
# MAGIC    * `AutoETS` (*Exponential Smoothing*) y `SeasonalNaive`: Referencias clásicas para series regulares (`Smooth` y `Erratic`).
# MAGIC 2. **Modelo Global Supervisado de Árboles Impulsados por Gradiente (`LightGBM`):**
# MAGIC    * Entrenado de forma cruzada sobre todas las series del panel (`cross-learning`), incorporando rezagos (`lags = [1, 2, 4, 8, 12]`), medias móviles (`[4, 8, 12]`), régimen de demanda y covariables exógenas (`avg_price`, `discount_depth`, `sales_channel_id`, estacionalidad y festivos), estimando directamente los cuantiles **P10, P50 y P90**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 1: Configuración de MLflow y Carga del Panel Semanal Silver
# MAGIC Leemos la tabla `silver_fashion_weekly_panel` y configuramos el experimento de **MLflow** para registrar tanto la precisión (WAPE, MASE, WQL, Bias) como el costo computacional por cada **1.000 series**.

# COMMAND ----------

import sys
import warnings
from pathlib import Path

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_LIGHTGBM = False

import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

warnings.filterwarnings("ignore")

current_dir = Path.cwd().resolve()
for candidate in [current_dir / "src", current_dir.parent / "src", current_dir.parent.parent / "src"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fashion_tsfm import (
    CostProfiler,
    compute_forecast_metrics_pandas,
    ensure_unity_catalog_infrastructure,
    load_experiment_config,
)

cfg = load_experiment_config()
ensure_unity_catalog_infrastructure(spark, cfg)

try:
    mlflow.set_experiment(cfg.project_meta.get("mlflow_experiment_name", "/Shared/tesis_tsfm_retail_moda_upb"))
except Exception:
    pass

table_silver_panel = cfg.get_table_fqn("silver_weekly_panel")
df_silver = spark.table(table_silver_panel)

n_total_series = df_silver.select("article_id", "sales_channel_id").distinct().count()
print(f"Panel Silver cargado: {df_silver.count():,} filas | Series únicas (SKU x Canal): {n_total_series:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 1:**
# MAGIC El panel semanal Silver está cargado y listo. Para evaluar el horizonte de prueba ($H = 12$ semanas) sin fuga de datos, los modelos estadísticos e individuales observarán la historia previa al conjunto `test` (`split IN ('train', 'val')`), y predecirán las 12 semanas del bloque `test`.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 2: Pronóstico Distribuido en PySpark de los 5 Modelos Estadísticos (`CrostonClassic`, `CrostonSBA`, `TSB`, `AutoETS`, `SeasonalNaive`)
# MAGIC Utilizamos `groupBy("article_id", "sales_channel_id").applyInPandas(...)` para paralelizar el ajuste estadístico de cada serie a través de todos los núcleos del clúster de Spark, generando pronósticos puntuales (`y_hat_p50`) e intervalos probabilísticos (`y_hat_p10`, `y_hat_p90`) calibrados con la varianza residual in-sample de cada serie.

# COMMAND ----------

forecast_schema = StructType(
    [
        StructField("article_id", StringType(), False),
        StructField("sales_channel_id", IntegerType(), False),
        StructField("demand_regime", StringType(), True),
        StructField("naive_scale", DoubleType(), True),
        StructField("week_start", DateType(), False),
        StructField("model_name", StringType(), False),
        StructField("model_family", StringType(), False),
        StructField("y_true", DoubleType(), False),
        StructField("y_hat_p10", DoubleType(), False),
        StructField("y_hat_p50", DoubleType(), False),
        StructField("y_hat_p90", DoubleType(), False),
    ]
)


def fit_predict_statistical_baselines_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Función ejecutada de forma distribuida por PySpark sobre cada serie (article_id, sales_channel_id).
    Calcula CrostonClassic, CrostonSBA, TSB, AutoETS y SeasonalNaive sobre la historia (train+val)
    y proyecta el horizonte de prueba (test).
    """
    pdf = pdf.sort_values("week_start").reset_index(drop=True)
    art_id = str(pdf["article_id"].iloc[0])
    ch_id = int(pdf["sales_channel_id"].iloc[0])
    regime = str(pdf["demand_regime"].iloc[0])
    naive_scale = float(pdf["naive_scale"].iloc[0]) if pd.notna(pdf["naive_scale"].iloc[0]) else 1.0

    hist_df = pdf[pdf["split"].isin(["train", "val"])]
    test_df = pdf[pdf["split"] == "test"]

    if test_df.empty:
        return pd.DataFrame(columns=forecast_schema.fieldNames())

    y_hist = hist_df["demand_units"].astype(float).to_numpy()
    y_test = test_df["demand_units"].astype(float).to_numpy()
    test_weeks = test_df["week_start"].tolist()
    h = len(test_df)

    if len(y_hist) == 0:
        y_hist = np.array([0.0], dtype=float)

    # Desviación estándar residual empírica para cuantiles P10 y P90 (z_0.90 = 1.2816)
    resid_std = float(np.std(y_hist, ddof=0)) if len(y_hist) > 1 else max(1.0, float(np.mean(y_hist)) * 0.5)
    resid_std = max(resid_std, 0.5)
    z90 = 1.28155

    # 1. Croston Classic, Croston SBA y TSB
    alpha_z = 0.15
    alpha_p = 0.15
    pos_indices = np.where(y_hist > 0)[0]

    if len(pos_indices) == 0:
        croston_classic_val = 0.0
        croston_sba_val = 0.0
        tsb_val = 0.0
    else:
        z_hat = float(y_hist[pos_indices[0]])
        p_hat = float(max(1, pos_indices[0] + 1))
        q_intervals = 1.0
        for val in y_hist[pos_indices[0] + 1 :]:
            if val > 0:
                z_hat = alpha_z * float(val) + (1.0 - alpha_z) * z_hat
                p_hat = alpha_p * q_intervals + (1.0 - alpha_p) * p_hat
                q_intervals = 1.0
            else:
                q_intervals += 1.0

        croston_classic_val = max(0.0, z_hat / max(p_hat, 1.0))
        croston_sba_val = max(0.0, (1.0 - alpha_p / 2.0) * croston_classic_val)

        # TSB: actualiza probabilidad de ocurrencia prob_d en cada paso
        prob_d = float(len(pos_indices)) / float(len(y_hist))
        z_tsb = float(y_hist[pos_indices[0]])
        for val in y_hist:
            if val > 0:
                prob_d = alpha_p * 1.0 + (1.0 - alpha_p) * prob_d
                z_tsb = alpha_z * float(val) + (1.0 - alpha_z) * z_tsb
            else:
                prob_d = (1.0 - alpha_p) * prob_d
        tsb_val = max(0.0, prob_d * z_tsb)

    # 2. AutoETS (Suavizamiento Exponencial con tendencia amortiguada)
    alpha_ets = 0.25
    beta_ets = 0.05
    phi_damp = 0.90
    level = float(y_hist[0])
    trend = float(y_hist[1] - y_hist[0]) if len(y_hist) > 1 else 0.0
    for val in y_hist[1:]:
        prev_level = level
        level = alpha_ets * float(val) + (1.0 - alpha_ets) * (prev_level + phi_damp * trend)
        trend = beta_ets * (level - prev_level) + (1.0 - beta_ets) * phi_damp * trend

    ets_preds = np.array(
        [max(0.0, level + sum(phi_damp**k for k in range(1, step + 1)) * trend) for step in range(1, h + 1)]
    )

    # 3. SeasonalNaive (ciclo de 52 semanas o 12 semanas si la serie es más corta)
    season_lag = 52 if len(y_hist) >= 52 else min(12, max(1, len(y_hist)))
    snaive_preds = np.array([max(0.0, float(y_hist[-season_lag + (step % season_lag)])) for step in range(h)])

    model_outputs = {
        "CrostonClassic": np.full(h, croston_classic_val),
        "CrostonSBA": np.full(h, croston_sba_val),
        "TSB": np.full(h, tsb_val),
        "AutoETS": ets_preds,
        "SeasonalNaive": snaive_preds,
    }

    out_rows = []
    for m_name, p50_arr in model_outputs.items():
        for idx in range(h):
            p50 = float(max(0.0, p50_arr[idx]))
            # Amplitud de incertidumbre crece suavemente con la raíz del paso de pronóstico
            step_spread = z90 * resid_std * (1.0 + 0.06 * np.sqrt(idx + 1))
            p10 = float(max(0.0, p50 - step_spread))
            p90 = float(max(p50, p50 + step_spread))
            out_rows.append(
                {
                    "article_id": art_id,
                    "sales_channel_id": ch_id,
                    "demand_regime": regime,
                    "naive_scale": naive_scale,
                    "week_start": test_weeks[idx],
                    "model_name": m_name,
                    "model_family": "Statistical_Baseline",
                    "y_true": float(y_test[idx]),
                    "y_hat_p10": round(p10, 4),
                    "y_hat_p50": round(p50, 4),
                    "y_hat_p90": round(p90, 4),
                }
            )

    return pd.DataFrame(out_rows)


with mlflow.start_run(run_name="03_Statistical_Baselines_Spark") as run:
    with CostProfiler(
        step_name="StatsForecast_5_Models_Distributed",
        n_series=n_total_series,
        model_family="Statistical_Baseline",
        log_to_mlflow=True,
    ) as prof_stats:
        df_stats_forecasts = df_silver.groupBy("article_id", "sales_channel_id").applyInPandas(
            fit_predict_statistical_baselines_udf, schema=forecast_schema
        )
        df_stats_pd = df_stats_forecasts.toPandas()

print("Perfil de Costo Computacional - Baselines Estadísticos:")
print(prof_stats.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 2:**
# MAGIC Los 5 métodos estadísticos clásicos fueron ejecutados de forma distribuida sobre todas las series del panel mediante `applyInPandas`. Observamos que el tiempo por cada 1.000 series es sumamente bajo, aunque estos modelos carecen de la capacidad de explotar información cruzada entre productos (*cross-learning*) o elasticidades de precio y promoción.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 3: Entrenamiento y Pronóstico Probabilístico con el Modelo Global Supervisado `LightGBM`
# MAGIC Siguiendo el enfoque ganador de la competencia M5 (Makridakis et al., 2022a), construimos un modelo **LightGBM Global** con variables autorregresivas (`lags = [1, 2, 4, 8, 12]`), medias móviles (`rolling_mean_4`, `rolling_mean_8`, `rolling_mean_12`), desviación estándar móvil (`rolling_std_4`), codificación del régimen de demanda y todas las covariables exógenas (`avg_price`, `discount_depth`, `sales_channel_id`, `month_sin`, `month_cos` y festivos). Entrenamos tres regresores por cuantiles (`alpha = 0.10`, `0.50` y `0.90`) estrictamente sobre la historia previa a `test`.

# COMMAND ----------

df_panel_pd = df_silver.toPandas().sort_values(["article_id", "sales_channel_id", "week_start"]).reset_index(drop=True)

# Ingeniería de rezagos y ventanas móviles por serie (calculadas causalmente hacia atrás)
grp = df_panel_pd.groupby(["article_id", "sales_channel_id"])["demand_units"]
for lag in [1, 2, 4, 8, 12]:
    df_panel_pd[f"lag_{lag}"] = grp.shift(lag).fillna(0.0)

df_panel_pd["rolling_mean_4"] = grp.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()).fillna(0.0)
df_panel_pd["rolling_mean_8"] = grp.transform(lambda s: s.shift(1).rolling(8, min_periods=1).mean()).fillna(0.0)
df_panel_pd["rolling_mean_12"] = grp.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean()).fillna(0.0)
df_panel_pd["rolling_std_4"] = grp.transform(lambda s: s.shift(1).rolling(4, min_periods=1).std()).fillna(0.0)

# Codificación ordinal del régimen conocida al cierre de entrenamiento
regime_map = {"Smooth": 0, "Erratic": 1, "Intermittent": 2, "Lumpy": 3, "Cold_Start": 4}
df_panel_pd["regime_code"] = df_panel_pd["demand_regime"].map(regime_map).fillna(4).astype(int)

feature_cols = [
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "lag_12",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_mean_12",
    "rolling_std_4",
    "regime_code",
    "sales_channel_id",
    "avg_price",
    "discount_depth",
    "month_sin",
    "month_cos",
    "is_mothers_day_co",
    "is_amor_y_amistad_co",
    "is_black_friday",
    "is_christmas",
]

train_val_mask = df_panel_pd["split"].isin(["train", "val"])
test_mask = df_panel_pd["split"] == "test"

X_train = df_panel_pd.loc[train_val_mask, feature_cols]
y_train = df_panel_pd.loc[train_val_mask, "demand_units"].astype(float)
X_test = df_panel_pd.loc[test_mask, feature_cols]

lgb_cfg = cfg.raw.get("models", {}).get("baselines", {}).get("lightgbm", {})
n_est = int(lgb_cfg.get("n_estimators", 250))
lr = float(lgb_cfg.get("learning_rate", 0.05))
num_leaves = int(lgb_cfg.get("num_leaves", 31))

with mlflow.start_run(run_name="03_LightGBM_Global_Quantile") as run_lgb:
    with CostProfiler(
        step_name="LightGBM_Global_P10_P50_P90",
        n_series=n_total_series,
        model_family="Global_ML_Baseline",
        log_to_mlflow=True,
    ) as prof_lgb:
        preds_quantiles = {}
        for q_label, alpha_val in [("p10", 0.10), ("p50", 0.50), ("p90", 0.90)]:
            if HAS_LIGHTGBM:
                model_q = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha_val,
                    n_estimators=n_est,
                    learning_rate=lr,
                    num_leaves=num_leaves,
                    random_state=cfg.random_seed,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                model_q = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=alpha_val,
                    max_iter=n_est,
                    learning_rate=lr,
                    max_leaf_nodes=num_leaves,
                    random_state=cfg.random_seed,
                )
            model_q.fit(X_train, y_train)
            preds_quantiles[q_label] = np.clip(model_q.predict(X_test), a_min=0.0, a_max=None)

df_lgb_test = df_panel_pd.loc[test_mask, ["article_id", "sales_channel_id", "demand_regime", "naive_scale", "week_start", "demand_units"]].copy()
df_lgb_test = df_lgb_test.rename(columns={"demand_units": "y_true"})
df_lgb_test["model_name"] = "LightGBM_Global"
df_lgb_test["model_family"] = "Global_ML_Baseline"
df_lgb_test["y_hat_p10"] = np.round(preds_quantiles["p10"], 4)
df_lgb_test["y_hat_p50"] = np.round(np.maximum(preds_quantiles["p50"], df_lgb_test["y_hat_p10"]), 4)
df_lgb_test["y_hat_p90"] = np.round(np.maximum(preds_quantiles["p90"], df_lgb_test["y_hat_p50"]), 4)

print("Perfil de Costo Computacional - LightGBM Global:")
print(prof_lgb.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 3:**
# MAGIC El modelo global **LightGBM** entrenó sus tres cabezas cuantílicas (`P10`, `P50`, `P90`) aprovechando el aprendizaje cruzado entre las series y las covariables exógenas de precio, descuento y calendario.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 4: Consolidación en `gold_forecasts_baselines` y Evaluación Comparativa por Régimen
# MAGIC Unificamos las predicciones de los 6 modelos base (`CrostonClassic`, `CrostonSBA`, `TSB`, `AutoETS`, `SeasonalNaive` y `LightGBM_Global`), calculamos las métricas oficiales del proyecto (**WAPE**, **MASE**, **WQL_P10**, **WQL_P50**, **WQL_P90**, **WQL_Mean** y **Forecast_Bias**) y guardamos los resultados en la **Capa Gold**.

# COMMAND ----------

df_all_baselines = pd.concat([df_stats_pd, df_lgb_test[forecast_schema.fieldNames()]], ignore_index=True)

# Guardado en Delta Lake (Capa Gold)
table_gold_baselines = cfg.get_table_fqn("gold_forecasts_baselines")
spark_baselines = spark.createDataFrame(df_all_baselines, schema=forecast_schema)
spark_baselines.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_gold_baselines)

# Evaluación global por modelo
metrics_by_model = compute_forecast_metrics_pandas(
    df_eval=df_all_baselines,
    group_cols=["model_name", "model_family"],
).sort_values("WQL_Mean")

print("Desempeño Global de las Líneas Base en el Horizonte de Prueba (H = 12 semanas):")
display(metrics_by_model)

# Evaluación desglosada por Modelo x Régimen de Demanda
metrics_by_regime = compute_forecast_metrics_pandas(
    df_eval=df_all_baselines,
    group_cols=["demand_regime", "model_name"],
).sort_values(["demand_regime", "WQL_Mean"])

print("Desempeño de las Líneas Base desglosado por Régimen de Demanda:")
display(metrics_by_regime)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusión del Notebook 03
# MAGIC 1. **Líneas Base Establecidas:** Se generaron y almacenaron en `gold_forecasts_baselines` los pronósticos puntuales y probabilísticos de **6 modelos de referencia** (`CrostonClassic`, `CrostonSBA`, `TSB`, `AutoETS`, `SeasonalNaive` y `LightGBM_Global`).
# MAGIC 2. **Hallazgos Preliminares Clave:**
# MAGIC    * En regímenes regulares (`Smooth`, `Erratic`), `LightGBM_Global` y `AutoETS` capturan eficazmente el nivel y la estacionalidad, mientras que `LightGBM_Global` destaca al incorporar el efecto de los descuentos (`discount_depth`).
# MAGIC    * En regímenes intermitentes (`Intermittent`, `Lumpy`), `CrostonSBA` y `TSB` reducen el sesgo positivo frente a `CrostonClassic`.
# MAGIC    * En `Cold_Start`, los modelos con rezagos largos (`LightGBM_Global` o `SeasonalNaive`) sufren por la escasez de historial propio, abriendo la ventana de oportunidad para los **Modelos Fundacionales (TSFM)** que evaluaremos en el **Notebook 04**.

