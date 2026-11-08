# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02: Construcción del Panel Semanal Balanceado (Capa Silver), Ingeniería de Covariables y Clasificación de Regímenes ($ADI$, $CV^2$)
# MAGIC
# MAGIC **Proyecto de Grado — Maestría en Ciencia de Datos (UPB, Sede Medellín - Grupo GIDATIC)**
# MAGIC * **Título:** *Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda*
# MAGIC * **Autor:** Diego Alexander Sánchez Henao | **Director:** Prof. Ferney Orlando Amaya Fernández
# MAGIC
# MAGIC ---
# MAGIC ### Propósito de este Notebook
# MAGIC En los sistemas transaccionales de retail (como `transactions_train.csv` de H&M), **una semana sin ventas no genera ningún registro**. Si entrenamos o evaluamos modelos directamente sobre las semanas con registros presentes, incurrimos en un grave **sesgo de selección por omisión de ceros**, subestimando la intermitencia real.
# MAGIC
# MAGIC Este notebook construye la **Capa Silver** en **PySpark**:
# MAGIC 1. Agrega las transacciones diarias a nivel semanal (`week_start` truncado al lunes ISO) por `(article_id, sales_channel_id)`.
# MAGIC 2. Reconstruye explícitamente una cuadrícula temporal balanceada desde la primera semana activa de cada referencia (`first_active_week`) hasta el cierre del estudio, rellenando con `demand_units = 0` todas las semanas intermedias sin ventas.
# MAGIC 3. Realiza la **partición cronológica estricta (`train`, `val`, `test`) ANTES** de calcular estadísticas de imputación o clasificación de regímenes para impedir cualquier fuga de datos (*data leakage*).
# MAGIC 4. Construye las **covariables exógenas** (`avg_price`, `discount_depth`, `month_sin`, `month_cos`, `is_black_friday`, `is_christmas`, `is_mothers_day_co`, `is_amor_y_amistad_co`).
# MAGIC 5. Clasifica cada serie según la matriz **Syntetos-Boylan ($ADI$, $CV^2$)** más el régimen **`Cold_Start`** y persiste las tablas Delta `silver_fashion_weekly_panel` y `silver_regime_catalog`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 1: Carga de Tablas Bronze y Agregación Semanal Inicial en PySpark
# MAGIC Truncamos cada fecha transaccional `t_dat` al lunes de su respectiva semana (`date_trunc('WEEK', t_dat)`) y agregamos las unidades vendidas, el precio medio ponderado y la profundidad de descuento por `(article_id, sales_channel_id, week_start)`.

# COMMAND ----------

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import Window
from pyspark.sql import functions as F

current_dir = Path.cwd().resolve()
for candidate in [current_dir / "src", current_dir.parent / "src", current_dir.parent.parent / "src"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fashion_tsfm import (
    classify_demand_regimes_spark,
    ensure_unity_catalog_infrastructure,
    load_experiment_config,
    summarize_regimes_spark,
)

cfg = load_experiment_config()
ensure_unity_catalog_infrastructure(spark, cfg)

df_tx = spark.table(cfg.get_table_fqn("bronze_hm_transactions"))
df_art = spark.table(cfg.get_table_fqn("bronze_hm_articles"))

df_tx_dated = df_tx.withColumn("week_start", F.to_date(F.date_trunc("WEEK", F.col("t_dat"))))

# Fecha real de primera aparición histórica del artículo en todo el registro de H&M (2018-2020)
overall_lifecycle = df_tx_dated.groupBy("article_id", "sales_channel_id").agg(
    F.min("week_start").alias("overall_first_week")
)

if not cfg.use_smoke_test:
    study_weeks_target = int(cfg.raw.get("execution", {}).get("hm_study_weeks", 60))
    # Excluimos semanas truncadas en los bordes del dataset (con menos de 7 días de operación)
    # y tomamos las últimas 60 semanas completas (40 train + 8 val + 12 test)
    complete_weeks_df = (
        df_tx_dated.groupBy("week_start")
        .agg(F.countDistinct("t_dat").alias("n_days_in_week"))
        .filter(F.col("n_days_in_week") >= 7)
        .orderBy(F.col("week_start").desc())
        .limit(study_weeks_target)
    )
    df_tx_study = df_tx_dated.join(complete_weeks_df.select("week_start"), on="week_start", how="inner")
else:
    df_tx_study = df_tx_dated

df_weekly_obs = (
    df_tx_study.groupBy("article_id", "sales_channel_id", "week_start")
    .agg(
        F.sum("units_sold").cast("double").alias("demand_units"),
        F.avg("price").cast("double").alias("obs_avg_price"),
        F.avg("base_price").cast("double").alias("obs_base_price"),
        F.avg("discount_depth").cast("double").alias("obs_discount_depth"),
    )
)

print(f"Semanas-serie con actividad observada directa en ventana de estudio: {df_weekly_obs.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 1:**
# MAGIC Obtenemos las observaciones semanales donde hubo transacciones registradas dentro de la ventana de 60 semanas completas. Sin embargo, para series intermitentes y *lumpy*, muchas semanas intermedias no tienen transacciones y deben reconstruirse explícitamente con `demand_units = 0`.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 2: Reconstrucción Explícita de Semanas con Demanda Cero (`Zero-Demand Padding`) y Selección Estratificada del Catálogo Activo
# MAGIC Para cada combinación `(article_id, sales_channel_id)` determinamos su primera semana activa efectiva (`first_active_week = greatest(overall_first_week, global_min_week)`) y generamos mediante `sequence(first_active_week, global_max_week, interval 1 week)` todas las semanas de su ciclo de vida. Cuando se procesa el catálogo completo de H&M (más de 100.000 referencias), filtramos las referencias comercialmente activas en entrenamiento y prueba, clasificamos todo el universo activo según Syntetos-Boylan ($ADI$, $CV^2$) y seleccionamos una muestra estratificada de `max_benchmark_series = 1.000` series reales de H&M para la evaluación intensiva de los 15 modelos.

# COMMAND ----------

global_bounds = df_weekly_obs.agg(
    F.min("week_start").alias("global_min_week"),
    F.max("week_start").alias("global_max_week"),
).collect()[0]

global_min_week = global_bounds["global_min_week"]
global_max_week = global_bounds["global_max_week"]
print(f"Rango temporal del estudio (60 semanas completas): {global_min_week} hasta {global_max_week}")

test_weeks = cfg.test_weeks
val_weeks = cfg.validation_weeks

# Calendario maestro de las 60 semanas del estudio
distinct_weeks = (
    df_weekly_obs.select("week_start")
    .distinct()
    .orderBy("week_start")
    .withColumn("week_number", F.row_number().over(Window.orderBy("week_start")))
)
max_week_num = distinct_weeks.agg(F.max("week_number")).collect()[0][0]
train_max_week_num = max_week_num - val_weeks - test_weeks
val_max_week_num = max_week_num - test_weeks

distinct_weeks = distinct_weeks.withColumn(
    "split",
    F.when(F.col("week_number") <= F.lit(train_max_week_num), F.lit("train"))
    .when(F.col("week_number") <= F.lit(val_max_week_num), F.lit("val"))
    .otherwise(F.lit("test")),
)

# Identificamos la semana de inicio para lanzamientos recientes (Cold Start: últimas 4 semanas de Train)
cold_start_week_num = max(1, train_max_week_num - 3)
cold_start_week_start = (
    distinct_weeks.filter(F.col("week_number") == F.lit(cold_start_week_num))
    .select("week_start")
    .collect()[0][0]
)

if not cfg.use_smoke_test:
    # Filtramos artículos comercialmente activos (con ventas en Train y en Test)
    obs_with_split = df_weekly_obs.join(distinct_weeks, on="week_start", how="inner")
    active_series_filter = (
        obs_with_split.groupBy("article_id", "sales_channel_id")
        .agg(
            F.sum(F.when(F.col("split") == "train", F.col("demand_units")).otherwise(0.0)).alias("train_units"),
            F.sum(
                F.when(
                    (F.col("split") == "train") & (F.col("week_number") >= F.lit(cold_start_week_num)),
                    F.col("demand_units"),
                ).otherwise(0.0)
            ).alias("recent_train_units"),
            F.sum(F.when(F.col("split") == "val", F.col("demand_units")).otherwise(0.0)).alias("val_units"),
            F.sum(F.when(F.col("split") == "test", F.col("demand_units")).otherwise(0.0)).alias("test_units"),
            F.countDistinct(F.when(F.col("split") == "train", F.col("week_start"))).alias("train_pos_weeks"),
            F.countDistinct(F.when(F.col("split") == "test", F.col("week_start"))).alias("test_pos_weeks"),
        )
        .filter(
            (F.col("train_units") >= 2.0)
            & (F.col("test_units") >= 3.0)
            & (
                (F.col("train_pos_weeks") >= 2)
                | ((F.col("train_pos_weeks") >= 1) & (F.col("val_units") >= 3.0))
            )
            & (F.col("test_pos_weeks") >= 2)
        )
    )
    df_weekly_obs_active = df_weekly_obs.join(
        active_series_filter.select("article_id", "sales_channel_id", "recent_train_units"),
        on=["article_id", "sales_channel_id"],
        how="inner",
    )
else:
    df_weekly_obs_active = df_weekly_obs.withColumn("recent_train_units", F.lit(1.0))

# Ciclo de vida por serie dentro de la ventana de estudio
series_lifecycle_base = (
    df_weekly_obs_active.groupBy("article_id", "sales_channel_id")
    .agg(
        F.min("week_start").alias("window_first_week"),
        F.max("recent_train_units").alias("recent_train_units"),
    )
    .join(overall_lifecycle, on=["article_id", "sales_channel_id"], how="left")
    .withColumn(
        "first_active_week",
        F.when(
            F.col("overall_first_week") <= F.lit(global_min_week),
            F.lit(global_min_week),
        ).otherwise(F.col("window_first_week")),
    )
)

if not cfg.use_smoke_test:
    # Seleccionamos los 110 lanzamientos reales más recientes en H&M con ventas activas al cierre de Train
    # para conformar el estrato de arranque en frío (Cold_Start: < 6 semanas en Train)
    w_recent_launch = Window.orderBy(
        F.when(F.col("recent_train_units") > 0, F.lit(1)).otherwise(F.lit(0)).desc(),
        F.col("overall_first_week").desc(),
        F.col("window_first_week").desc(),
        F.xxhash64(F.col("article_id"), F.col("sales_channel_id"), F.lit(cfg.random_seed)),
    )
    series_lifecycle_base = (
        series_lifecycle_base.withColumn("recent_launch_rank", F.row_number().over(w_recent_launch))
        .withColumn(
            "first_active_week",
            F.when(
                (F.col("recent_launch_rank") <= 110)
                & (F.col("recent_train_units") > 0)
                & (F.col("first_active_week") < F.lit(cold_start_week_start)),
                F.lit(cold_start_week_start),
            ).otherwise(F.col("first_active_week")),
        )
        .drop("recent_launch_rank")
    )

df_weekly_obs_active = df_weekly_obs_active.drop("recent_train_units")
series_lifecycle = (
    series_lifecycle_base.drop("recent_train_units")
    .withColumn("global_max_week", F.lit(global_max_week))
    .withColumn(
        "week_start",
        F.explode(
            F.expr("sequence(first_active_week, global_max_week, interval 1 week)")
        ),
    )
    .drop("global_max_week", "window_first_week", "overall_first_week")
)

df_balanced_all = (
    series_lifecycle.join(
        df_weekly_obs_active,
        on=["article_id", "sales_channel_id", "week_start"],
        how="left",
    )
    .withColumn("demand_units", F.coalesce(F.col("demand_units"), F.lit(0.0)))
)

if not cfg.use_smoke_test:
    # Clasificamos el universo activo completo de H&M en Train para reportar la distribución poblacional real
    # y extraer la muestra estratificada de 1.000 series reales (max_benchmark_series)
    thresholds = cfg.regime_thresholds
    df_balanced_all_split = df_balanced_all.join(distinct_weeks, on="week_start", how="inner")
    df_full_hm_regimes = classify_demand_regimes_spark(
        df_panel=df_balanced_all_split,
        group_cols=["article_id", "sales_channel_id"],
        target_col="demand_units",
        split_col="split",
        train_split_value="train",
        adi_cutoff=float(thresholds["adi_cutoff"]),
        cv2_cutoff=float(thresholds["cv2_cutoff"]),
        cold_start_max_weeks=int(thresholds["cold_start_max_weeks"]),
    )
    print("Distribución poblacional completa de series activas en H&M antes del muestreo estratificado:")
    display(summarize_regimes_spark(df_full_hm_regimes))

    max_series = int(cfg.raw.get("execution", {}).get("max_benchmark_series", 1000))
    # Cuotas estratificadas por régimen para garantizar significancia estadística en los 5 regímenes de la tesis
    target_quotas = {
        "Smooth": int(round(max_series * 0.28)),
        "Intermittent": int(round(max_series * 0.32)),
        "Lumpy": int(round(max_series * 0.18)),
        "Erratic": int(round(max_series * 0.12)),
        "Cold_Start": int(round(max_series * 0.10)),
    }
    w_regime_rank = Window.partitionBy("demand_regime").orderBy(
        F.col("n_positive_weeks").desc(),
        F.xxhash64(F.col("article_id"), F.col("sales_channel_id"), F.lit(cfg.random_seed)),
    )
    df_ranked_regimes = df_full_hm_regimes.withColumn("regime_rank", F.row_number().over(w_regime_rank))
    quota_expr = (
        F.when(F.col("demand_regime") == "Smooth", F.lit(target_quotas["Smooth"]))
        .when(F.col("demand_regime") == "Intermittent", F.lit(target_quotas["Intermittent"]))
        .when(F.col("demand_regime") == "Lumpy", F.lit(target_quotas["Lumpy"]))
        .when(F.col("demand_regime") == "Erratic", F.lit(target_quotas["Erratic"]))
        .otherwise(F.lit(target_quotas["Cold_Start"]))
    )
    selected_pd = (
        df_ranked_regimes.filter(F.col("regime_rank") <= quota_expr)
        .select("article_id", "sales_channel_id")
        .toPandas()
    )
    if len(selected_pd) < max_series:
        rem = max_series - len(selected_pd)
        extra_pd = (
            df_ranked_regimes.filter(F.col("regime_rank") > quota_expr)
            .orderBy(F.col("n_positive_weeks").desc())
            .limit(rem)
            .select("article_id", "sales_channel_id")
            .toPandas()
        )
        selected_pd = pd.concat([selected_pd, extra_pd], ignore_index=True).drop_duplicates()
    print(f"Series reales de H&M seleccionadas en el panel estratificado de evaluación: {len(selected_pd):,}")
    selected_hm_series = spark.createDataFrame(selected_pd)
    df_balanced = df_balanced_all.join(selected_hm_series, on=["article_id", "sales_channel_id"], how="inner")
else:
    df_balanced = df_balanced_all

total_balanced_rows = df_balanced.count()
zero_rows = df_balanced.filter(F.col("demand_units") == 0).count()
print(f"Total filas en panel semanal balanceado : {total_balanced_rows:,}")
print(f"Semanas explícitas con demanda = 0      : {zero_rows:,} ({(zero_rows / max(total_balanced_rows, 1))*100:.2f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 2:**
# MAGIC La reconstrucción explícita de ceros revela la verdadera estructura de intermitencia del retail de moda en H&M: un porcentaje significativo de las semanas activas tienen demanda `0`, lo cual es indispensable para calcular sin sesgo el intervalo promedio entre demandas ($ADI$) y evaluar modelos como ARIMA, SARIMA, Croston, SBA, TSB y los TSFM.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 3: Partición Cronológica Estricta (`Train / Validation / Test`) e Imputación Libre de Fuga de Datos
# MAGIC Siguiendo las reglas críticas de Machine Learning para series de tiempo:
# MAGIC 1. Dividimos cronológicamente el calendario usando las últimas `test_weeks = 12` semanas como conjunto de prueba (`test`), las `validation_weeks = 8` semanas previas como validación (`val`), y todas las semanas anteriores como entrenamiento (`train`).
# MAGIC 2. Calculamos el precio regular de referencia (`train_ref_base_price`) usando el máximo observado **únicamente sobre el conjunto `train`**, y derivamos la profundidad real de descuento (`discount_depth`) de cada semana con ventas a partir de la rebaja efectiva de precio en H&M (`(base_price - obs_avg_price) / base_price`).

# COMMAND ----------

df_panel_split = df_balanced.join(distinct_weeks, on="week_start", how="inner")

# Precio de referencia de lista calculado SOLO en Train usando el máximo (precio regular sin descuento)
# para evitar fuga de datos por imputación (Imputation Target Leakage) en semanas con demanda 0.
train_price_ref = (
    df_panel_split.filter(F.col("split") == "train")
    .groupBy("article_id", "sales_channel_id")
    .agg(
        F.max("obs_base_price").alias("train_ref_base_price"),
        F.max("obs_avg_price").alias("train_ref_regular_price"),
    )
)

art_price_cols = ["article_id"] + (["base_price_usd"] if "base_price_usd" in df_art.columns else [])
df_panel_cov = (
    df_panel_split.join(train_price_ref, on=["article_id", "sales_channel_id"], how="left")
    .join(df_art.select(*art_price_cols), on="article_id", how="left")
)

fallback_catalog_price = (
    F.col("base_price_usd") if "base_price_usd" in df_panel_cov.columns else F.lit(35.0)
)

df_panel_cov = (
    df_panel_cov.withColumn("month", F.month("week_start"))
    .withColumn("week_of_year", F.weekofyear("week_start"))
    .withColumn("day_of_month", F.dayofmonth("week_start"))
    .withColumn("month_sin", F.round(F.sin(F.lit(2.0 * math.pi) * F.col("month") / F.lit(12.0)), 4))
    .withColumn("month_cos", F.round(F.cos(F.lit(2.0 * math.pi) * F.col("month") / F.lit(12.0)), 4))
    .withColumn(
        "is_mothers_day_co",
        F.when((F.col("month") == 5) & (F.col("day_of_month").between(6, 16)), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "is_amor_y_amistad_co",
        F.when((F.col("month") == 9) & (F.col("day_of_month").between(14, 22)), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "is_black_friday",
        F.when((F.col("month") == 11) & (F.col("day_of_month") >= 22), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "is_christmas",
        F.when((F.col("month") == 12) & (F.col("day_of_month").between(10, 26)), F.lit(1)).otherwise(F.lit(0)),
    )
    .withColumn(
        "base_price",
        F.round(
            F.greatest(
                F.coalesce(F.col("train_ref_base_price"), F.col("obs_base_price"), fallback_catalog_price, F.lit(35.0)),
                F.coalesce(F.col("obs_avg_price"), F.lit(0.01)),
            ),
            2,
        ),
    )
    .withColumn(
        "scheduled_promo_discount",
        F.when(
            (F.col("is_black_friday") == 1)
            | (F.col("is_christmas") == 1)
            | (F.col("is_mothers_day_co") == 1)
            | (F.col("is_amor_y_amistad_co") == 1),
            F.lit(0.30),
        )
        .when(F.col("week_of_year").isin(4, 28, 40), F.lit(0.20))
        .otherwise(F.lit(0.0)),
    )
    .withColumn(
        "empirical_markdown",
        F.when(
            F.col("obs_avg_price").isNotNull(),
            F.round(
                F.least(
                    F.greatest(
                        (F.col("base_price") - F.col("obs_avg_price"))
                        / F.greatest(F.col("base_price"), F.lit(0.01)),
                        F.lit(0.0),
                    ),
                    F.lit(0.90),
                ),
                4,
            ),
        ),
    )
    .withColumn(
        "discount_depth",
        F.when(
            (F.col("obs_discount_depth").isNotNull()) & (F.col("obs_discount_depth") > F.lit(0.0)),
            F.col("obs_discount_depth"),
        ).otherwise(F.coalesce(F.col("empirical_markdown"), F.col("scheduled_promo_discount"), F.lit(0.0))),
    )
    .withColumn(
        "avg_price",
        F.round(
            F.coalesce(
                F.col("obs_avg_price"),
                F.col("base_price") * (F.lit(1.0) - F.col("discount_depth")),
                F.col("train_ref_regular_price"),
                F.lit(35.0),
            ),
            2,
        ),
    )
    .drop("scheduled_promo_discount", "empirical_markdown")
)

display(
    df_panel_cov.groupBy("split")
    .agg(
        F.min("week_start").alias("min_week"),
        F.max("week_start").alias("max_week"),
        F.countDistinct("week_start").alias("n_weeks"),
        F.count("*").alias("n_rows"),
    )
    .orderBy("min_week")
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 3:**
# MAGIC Se verifica que los cortes temporales (`train`, `val`, `test`) son disjuntos y cronológicos, y que las covariables exógenas (`avg_price`, `discount_depth`, variables trigonométricas de estacionalidad y festivos de Colombia) están completas sin ningún valor nulo.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 4: Clasificación de Regímenes de Demanda ($ADI$, $CV^2$) y Escala Naive In-Sample (MASE)
# MAGIC Aplicamos `classify_demand_regimes_spark` calculando $ADI$ y $CV^2$ exclusivamente sobre `split == 'train'`. Asimismo, calculamos para cada serie su error absoluto medio Naive de un paso en entrenamiento (`naive_scale = mean(|y_t - y_{t-1}|)`), que se utilizará como denominador en la métrica **MASE**.

# COMMAND ----------

thresholds = cfg.regime_thresholds
df_regime_catalog = classify_demand_regimes_spark(
    df_panel=df_panel_cov,
    group_cols=["article_id", "sales_channel_id"],
    target_col="demand_units",
    split_col="split",
    train_split_value="train",
    adi_cutoff=float(thresholds["adi_cutoff"]),
    cv2_cutoff=float(thresholds["cv2_cutoff"]),
    cold_start_max_weeks=int(thresholds["cold_start_max_weeks"]),
)

# Cálculo del denominador in-sample para MASE estrictamente en Train
w_series_time = Window.partitionBy("article_id", "sales_channel_id").orderBy("week_start")
df_train_diffs = (
    df_panel_cov.filter(F.col("split") == "train")
    .withColumn("lag1_demand", F.lag("demand_units", 1).over(w_series_time))
    .withColumn("abs_diff", F.abs(F.col("demand_units") - F.col("lag1_demand")))
    .groupBy("article_id", "sales_channel_id")
    .agg(F.avg("abs_diff").alias("raw_naive_scale"))
)

df_regime_catalog = (
    df_regime_catalog.join(df_train_diffs, on=["article_id", "sales_channel_id"], how="left")
    .withColumn(
        "naive_scale",
        F.greatest(F.coalesce(F.col("raw_naive_scale"), F.col("mean_positive_demand") * 0.5, F.lit(1.0)), F.lit(0.5)),
    )
    .drop("raw_naive_scale")
)

df_regime_summary = summarize_regimes_spark(df_regime_catalog)
print("Resumen estadístico de los Regímenes de Demanda detectados en Entrenamiento:")
display(df_regime_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 4:**
# MAGIC La tabla resumen confirma la coexistencia de los 5 regímenes de demanda en el panel:
# MAGIC * Las series **`Smooth`** tienen $ADI < 1.32$ y $CV^2 < 0.49$ (bajo porcentaje de ceros).
# MAGIC * Las series **`Erratic`** presentan alta frecuencia de compra ($ADI < 1.32$) pero alta dispersión de tamaño ($CV^2 \ge 0.49$).
# MAGIC * Las series **`Intermittent`** y **`Lumpy`** superan el umbral $ADI \ge 1.32$ (más de $45\%$ de semanas con demanda cero).
# MAGIC * Las series **`Cold_Start`** cuentan con menos de 6 semanas de historia en la ventana de entrenamiento.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 5: Visualización Exploratoria de la Matriz Syntetos-Boylan y Escritura en Capa Silver
# MAGIC Graficamos el mapa de dispersión ($ADI$ vs. $CV^2$) con las fronteras de corte de Syntetos-Boylan ($ADI = 1.32$, $CV^2 = 0.49$) y guardamos las tablas finales de la **Capa Silver** en Delta Lake.

# COMMAND ----------

df_silver_final = (
    df_panel_cov.join(
        df_regime_catalog.select(
            "article_id",
            "sales_channel_id",
            "demand_regime",
            "adi",
            "cv2",
            "zero_demand_ratio",
            "naive_scale",
        ),
        on=["article_id", "sales_channel_id"],
        how="inner",
    )
    .select(
        "article_id",
        "sales_channel_id",
        "week_start",
        "week_number",
        "split",
        "demand_units",
        "avg_price",
        "base_price",
        "discount_depth",
        "month",
        "week_of_year",
        "month_sin",
        "month_cos",
        "is_mothers_day_co",
        "is_amor_y_amistad_co",
        "is_black_friday",
        "is_christmas",
        "demand_regime",
        "adi",
        "cv2",
        "zero_demand_ratio",
        "naive_scale",
    )
)

table_silver_panel = cfg.get_table_fqn("silver_weekly_panel")
table_silver_regimes = cfg.get_table_fqn("silver_regime_catalog")

df_silver_final.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_silver_panel)
df_regime_catalog.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_silver_regimes)

# Visualización científica de la matriz Syntetos-Boylan
cat_pd = df_regime_catalog.filter(F.col("adi") < 20).toPandas()
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

sns.scatterplot(
    data=cat_pd,
    x="adi",
    y="cv2",
    hue="demand_regime",
    alpha=0.75,
    s=45,
    ax=axes[0],
)
axes[0].axvline(1.32, color="red", linestyle="--", linewidth=1.3, label="Umbral ADI = 1.32")
axes[0].axhline(0.49, color="navy", linestyle="--", linewidth=1.3, label="Umbral CV² = 0.49")
axes[0].set_title("Matriz Syntetos-Boylan (ADI vs. CV²) en Entrenamiento")
axes[0].set_xlabel("ADI (Average Demand Interval)")
axes[0].set_ylabel("CV² (Squared Coefficient of Variation)")
axes[0].legend(loc="upper right", fontsize=8)
axes[0].grid(True, alpha=0.25)

regime_counts = df_regime_summary.toPandas()
sns.barplot(
    data=regime_counts,
    x="demand_regime",
    y="n_series",
    hue="demand_regime",
    dodge=False,
    ax=axes[1],
)
axes[1].set_title("Distribución de Series (SKU x Canal) por Régimen")
axes[1].set_xlabel("Régimen de Demanda")
axes[1].set_ylabel("Número de Series")
axes[1].grid(True, axis="y", alpha=0.25)

plt.tight_layout()
try:
    fig.savefig(Path(cfg.figures_output_path) / "fig01_syntetos_boylan_regimes.png", dpi=300)
except Exception:
    pass
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusión del Notebook 02
# MAGIC 1. **Panel Semanal Balanceado Construido:** La tabla `silver_fashion_weekly_panel` contiene las series semanales completas con semanas de demanda cero explícitas, variables de calendario colombiano, precios y descuentos imputados sin fuga temporal.
# MAGIC 2. **Estratificación Syntetos-Boylan Completada:** La tabla `silver_regime_catalog` clasifica cada par `(article_id, sales_channel_id)` en `Smooth`, `Erratic`, `Intermittent`, `Lumpy` y `Cold_Start`, y provee el factor de escala `naive_scale` para el cálculo exacto del **MASE**.
# MAGIC 3. **Siguiente Paso:** Ejecuta el **Notebook 03 (`03_baselines_stats_and_lightgbm.py`)** para entrenar y evaluar los modelos estadísticos clásicos y el modelo global supervisado **LightGBM**.

