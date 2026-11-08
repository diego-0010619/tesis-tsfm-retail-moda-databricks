"""
Clasificador de Regímenes de Demanda según la Matriz Syntetos-Boylan (2005)
extendida con el régimen de Arranque en Frío (`Cold_Start`) para Retail de Moda.

Definiciones matemáticas (calculadas estrictamente sobre la ventana de entrenamiento
para evitar fuga de información temporal / data leakage):
- N_active : Número total de semanas activas de la serie en la ventana de entrenamiento.
- N_pos    : Número de semanas con demanda estrictamente positiva (y_t > 0).
- ADI      : Average Demand Interval = N_active / N_pos.
- CV^2     : Squared Coefficient of Variation de las demandas positivas = (sigma_+ / mu_+)^2.

Reglas de Clasificación:
1. `Cold_Start`   : N_active < 6 semanas (o N_pos == 0 en entrenamiento).
2. `Smooth`       : ADI < 1.32  y  CV^2 < 0.49
3. `Erratic`      : ADI < 1.32  y  CV^2 >= 0.49
4. `Intermittent` : ADI >= 1.32 y  CV^2 < 0.49
5. `Lumpy`        : ADI >= 1.32 y  CV^2 >= 0.49
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd


def classify_demand_regimes_spark(
    df_panel: Any,
    group_cols: Optional[List[str]] = None,
    target_col: str = "demand_units",
    split_col: Optional[str] = "split",
    train_split_value: str = "train",
    adi_cutoff: float = 1.32,
    cv2_cutoff: float = 0.49,
    cold_start_max_weeks: int = 6,
) -> Any:
    """
    Calcula en PySpark el ADI, CV^2 y el régimen de demanda para cada serie
    identificada por `group_cols` (por defecto `['article_id', 'sales_channel_id']`).

    Importante (Prevención de Data Leakage):
    Si `split_col` existe en `df_panel`, las estadísticas ADI y CV^2 se calculan
    exclusivamente sobre `split == train_split_value`. Si una serie aparece por
    primera vez en validación o test (0 semanas en train), se clasifica automáticamente
    como `Cold_Start`.

    Parameters
    ----------
    df_panel : pyspark.sql.DataFrame
        Panel semanal con columnas de agrupación, demanda y partición temporal.
    group_cols : List[str], optional
        Columnas que identifican unívocamente cada serie de tiempo.
    target_col : str
        Nombre de la columna de demanda semanal.
    split_col : str, optional
        Columna que indica la partición temporal ('train', 'val', 'test').
    train_split_value : str
        Valor que identifica el conjunto de entrenamiento.
    adi_cutoff : float
        Umbral Syntetos-Boylan para ADI (por defecto 1.32).
    cv2_cutoff : float
        Umbral Syntetos-Boylan para CV^2 (por defecto 0.49).
    cold_start_max_weeks : int
        Umbral máximo de semanas activas en entrenamiento para considerar Cold Start (6).

    Returns
    -------
    pyspark.sql.DataFrame
        Catálogo por serie con columnas:
        `[*group_cols, 'n_train_weeks', 'n_positive_weeks', 'zero_demand_ratio',
          'mean_positive_demand', 'std_positive_demand', 'adi', 'cv2', 'demand_regime']`
    """
    from pyspark.sql import functions as F

    if group_cols is None:
        group_cols = ["article_id", "sales_channel_id"]

    # Obtenemos el universo completo de series presentes en el panel
    distinct_series = df_panel.select(*group_cols).distinct()

    # Filtramos estrictamente al periodo de entrenamiento si existe la columna split
    if split_col and split_col in df_panel.columns:
        df_train = df_panel.filter(F.col(split_col) == F.lit(train_split_value))
    else:
        df_train = df_panel

    # Expresión para demanda estrictamente positiva (NULL cuando es 0 para que avg/stddev
    # ignoren las semanas en cero al calcular mu_+ y sigma_+)
    pos_demand_expr = F.when(F.col(target_col) > 0, F.col(target_col).cast("double")).otherwise(
        F.lit(None)
    )

    stats_df = df_train.groupBy(*group_cols).agg(
        F.count(F.lit(1)).cast("int").alias("n_train_weeks"),
        F.sum(F.when(F.col(target_col) > 0, 1).otherwise(0)).cast("int").alias("n_positive_weeks"),
        F.avg(pos_demand_expr).alias("mean_positive_demand"),
        # Usamos stddev_pop o stddev_samp con fallback a 0 cuando solo hay 1 observación positiva
        F.coalesce(F.stddev_pop(pos_demand_expr), F.lit(0.0)).alias("std_positive_demand"),
        F.sum(F.col(target_col)).cast("double").alias("total_train_units"),
    )

    # Unimos con el universo de series por si alguna serie nació después del corte de train
    full_stats = (
        distinct_series.join(stats_df, on=group_cols, how="left")
        .fillna(
            {
                "n_train_weeks": 0,
                "n_positive_weeks": 0,
                "mean_positive_demand": 0.0,
                "std_positive_demand": 0.0,
                "total_train_units": 0.0,
            }
        )
        .withColumn(
            "zero_demand_ratio",
            F.when(
                F.col("n_train_weeks") > 0,
                F.lit(1.0) - (F.col("n_positive_weeks") / F.col("n_train_weeks")),
            ).otherwise(F.lit(1.0)),
        )
        .withColumn(
            "adi",
            F.when(
                F.col("n_positive_weeks") > 0,
                F.col("n_train_weeks").cast("double") / F.col("n_positive_weeks").cast("double"),
            ).otherwise(F.lit(999.0)),
        )
        .withColumn(
            "cv2",
            F.when(
                F.col("mean_positive_demand") > 1e-9,
                F.pow(F.col("std_positive_demand") / F.col("mean_positive_demand"), 2.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "demand_regime",
            F.when(
                (F.col("n_train_weeks") < F.lit(cold_start_max_weeks))
                | (F.col("n_positive_weeks") == 0),
                F.lit("Cold_Start"),
            )
            .when(
                (F.col("adi") < F.lit(adi_cutoff)) & (F.col("cv2") < F.lit(cv2_cutoff)),
                F.lit("Smooth"),
            )
            .when(
                (F.col("adi") < F.lit(adi_cutoff)) & (F.col("cv2") >= F.lit(cv2_cutoff)),
                F.lit("Erratic"),
            )
            .when(
                (F.col("adi") >= F.lit(adi_cutoff)) & (F.col("cv2") < F.lit(cv2_cutoff)),
                F.lit("Intermittent"),
            )
            .otherwise(F.lit("Lumpy")),
        )
    )

    return full_stats


def summarize_regimes_spark(df_regime_catalog: Any) -> Any:
    """
    Genera una tabla resumen agregada en PySpark con el conteo de series,
    porcentaje de participación y promedios de ADI, CV^2 y proporción de ceros
    por cada régimen de demanda.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    summary = df_regime_catalog.groupBy("demand_regime").agg(
        F.count(F.lit(1)).alias("n_series"),
        F.round(F.avg("n_train_weeks"), 2).alias("avg_train_weeks"),
        F.round(F.avg("zero_demand_ratio") * 100.0, 2).alias("avg_zero_pct"),
        F.round(F.avg(F.when(F.col("adi") < 900, F.col("adi"))), 3).alias("avg_adi"),
        F.round(F.avg("cv2"), 3).alias("avg_cv2"),
        F.round(F.avg("mean_positive_demand"), 2).alias("avg_pos_demand"),
    )

    w_all = Window.partitionBy()
    summary = summary.withColumn(
        "share_pct", F.round((F.col("n_series") / F.sum("n_series").over(w_all)) * 100.0, 2)
    ).orderBy(F.desc("n_series"))

    return summary


def classify_demand_regimes_pandas(
    df_panel: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    target_col: str = "demand_units",
    split_col: Optional[str] = "split",
    train_split_value: str = "train",
    adi_cutoff: float = 1.32,
    cv2_cutoff: float = 0.49,
    cold_start_max_weeks: int = 6,
) -> pd.DataFrame:
    """
    Versión vectorizada en Pandas del clasificador Syntetos-Boylan (2005) + `Cold_Start`.
    Útil para validaciones locales, pruebas unitarias o sub-muestras en memoria.
    """
    if group_cols is None:
        group_cols = ["article_id", "sales_channel_id"]

    distinct_series = df_panel[group_cols].drop_duplicates().copy()

    if split_col and split_col in df_panel.columns:
        df_train = df_panel[df_panel[split_col] == train_split_value].copy()
    else:
        df_train = df_panel.copy()

    if df_train.empty:
        distinct_series["n_train_weeks"] = 0
        distinct_series["n_positive_weeks"] = 0
        distinct_series["zero_demand_ratio"] = 1.0
        distinct_series["mean_positive_demand"] = 0.0
        distinct_series["std_positive_demand"] = 0.0
        distinct_series["adi"] = 999.0
        distinct_series["cv2"] = 0.0
        distinct_series["demand_regime"] = "Cold_Start"
        return distinct_series

    df_train["_is_pos"] = (df_train[target_col] > 0).astype(int)
    df_train["_pos_val"] = np.where(df_train[target_col] > 0, df_train[target_col], np.nan)

    grouped = df_train.groupby(group_cols, as_index=False).agg(
        n_train_weeks=(target_col, "count"),
        n_positive_weeks=("_is_pos", "sum"),
        mean_positive_demand=("_pos_val", "mean"),
        std_positive_demand=("_pos_val", lambda s: float(np.nanstd(s, ddof=0)) if s.notna().any() else 0.0),
        total_train_units=(target_col, "sum"),
    )

    res = distinct_series.merge(grouped, on=group_cols, how="left")
    res["n_train_weeks"] = res["n_train_weeks"].fillna(0).astype(int)
    res["n_positive_weeks"] = res["n_positive_weeks"].fillna(0).astype(int)
    res["mean_positive_demand"] = res["mean_positive_demand"].fillna(0.0)
    res["std_positive_demand"] = res["std_positive_demand"].fillna(0.0)
    res["total_train_units"] = res["total_train_units"].fillna(0.0)

    res["zero_demand_ratio"] = np.where(
        res["n_train_weeks"] > 0,
        1.0 - (res["n_positive_weeks"] / res["n_train_weeks"]),
        1.0,
    )
    res["adi"] = np.where(
        res["n_positive_weeks"] > 0,
        res["n_train_weeks"] / res["n_positive_weeks"],
        999.0,
    )
    res["cv2"] = np.where(
        res["mean_positive_demand"] > 1e-9,
        (res["std_positive_demand"] / res["mean_positive_demand"]) ** 2,
        0.0,
    )

    conditions = [
        (res["n_train_weeks"] < cold_start_max_weeks) | (res["n_positive_weeks"] == 0),
        (res["adi"] < adi_cutoff) & (res["cv2"] < cv2_cutoff),
        (res["adi"] < adi_cutoff) & (res["cv2"] >= cv2_cutoff),
        (res["adi"] >= adi_cutoff) & (res["cv2"] < cv2_cutoff),
    ]
    choices = ["Cold_Start", "Smooth", "Erratic", "Intermittent"]
    res["demand_regime"] = np.select(conditions, choices, default="Lumpy")

    return res


def summarize_regimes_pandas(df_regime_catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Resumen estadístico en Pandas por régimen de demanda.
    """
    total_series = max(len(df_regime_catalog), 1)
    df_valid_adi = df_regime_catalog.copy()
    df_valid_adi["_valid_adi"] = np.where(df_valid_adi["adi"] < 900, df_valid_adi["adi"], np.nan)

    summary = (
        df_valid_adi.groupby("demand_regime", as_index=False)
        .agg(
            n_series=("demand_regime", "count"),
            avg_train_weeks=("n_train_weeks", "mean"),
            avg_zero_pct=("zero_demand_ratio", lambda s: float(np.mean(s) * 100.0)),
            avg_adi=("_valid_adi", "mean"),
            avg_cv2=("cv2", "mean"),
            avg_pos_demand=("mean_positive_demand", "mean"),
        )
        .sort_values("n_series", ascending=False)
    )
    summary["share_pct"] = np.round((summary["n_series"] / total_series) * 100.0, 2)
    return summary.round(3)

