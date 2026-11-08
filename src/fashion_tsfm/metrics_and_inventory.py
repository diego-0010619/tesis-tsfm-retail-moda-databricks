"""
Módulo de Métricas de Pronóstico (Puntuales y Probabilísticas) y Simulación
Operativa de Inventario orientada a decisiones (*Decision-Aware Evaluation*).

Implementa las métricas oficiales del Marco PICO de la Tesis:
1. Precisión puntual y probabilística:
   - WAPE (*Weighted Absolute Percentage Error*)
   - MASE (*Mean Absolute Scaled Error*)
   - WQL (*Weighted Quantile Loss* en cuantiles P10, P50, P90 y promedio)
   - Forecast Bias (*Sesgo porcentual de pronóstico*)
2. Impacto en decisiones de inventario (Política periódica (R, S) con Lead Time):
   - Fill Rate real alcanzado (nivel de servicio volumétrico)
   - Unidades en quiebre de inventario (*Lost Sales*)
   - Exceso de inventario promedio inmovilizado (*Overstock / Average On-Hand*)
   - Curva de eficiencia Nivel de Servicio vs. Stock Promedio
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _pinball_loss(y_true: np.ndarray, y_quant: np.ndarray, q: float) -> np.ndarray:
    """
    Calcula la pérdida Pinball (Quantile Loss) elemento a elemento:
    QL_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    Para mantener la convención estándar de WQL (2 * sum(QL) / sum(|y|)),
    multiplicamos por 2 de modo que WQL(P50) coincida exactamente con WAPE.
    """
    err = y_true - y_quant
    return 2.0 * np.maximum(q * err, (q - 1.0) * err)


def compute_forecast_metrics_pandas(
    df_eval: pd.DataFrame,
    df_train_scale: Optional[pd.DataFrame] = None,
    group_cols: Optional[List[str]] = None,
    actual_col: str = "y_true",
    pred_p50_col: str = "y_hat_p50",
    pred_p10_col: str = "y_hat_p10",
    pred_p90_col: str = "y_hat_p90",
) -> pd.DataFrame:
    """
    Calcula WAPE, MASE, WQL (P10, P50, P90 y Mean WQL) y Forecast Bias desglosado
    por las columnas indicadas en `group_cols` (ej. `['model_name', 'demand_regime']`).

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame con valores reales y pronósticos en el horizonte de prueba.
    df_train_scale : pd.DataFrame, optional
        DataFrame con columna `naive_scale` por `(article_id, sales_channel_id)`
        calculada sobre el conjunto de entrenamiento para el denominador del MASE.
    group_cols : List[str], optional
        Columnas de agrupación para el reporte (ej. `['model_name', 'demand_regime']`).
    """
    if group_cols is None:
        group_cols = ["model_name", "demand_regime"]

    df = df_eval.copy()
    if df_train_scale is not None and "naive_scale" not in df.columns:
        id_cols = [c for c in ["article_id", "sales_channel_id"] if c in df.columns and c in df_train_scale.columns]
        if id_cols:
            df = df.merge(df_train_scale[id_cols + ["naive_scale"]], on=id_cols, how="left")

    if "naive_scale" not in df.columns:
        df["naive_scale"] = np.nan

    y_true = df[actual_col].astype(float).to_numpy()
    y_p50 = np.clip(df[pred_p50_col].astype(float).to_numpy(), a_min=0.0, a_max=None)
    y_p10 = (
        np.clip(df[pred_p10_col].astype(float).to_numpy(), a_min=0.0, a_max=None)
        if pred_p10_col in df.columns
        else y_p50 * 0.65
    )
    y_p90 = (
        np.clip(df[pred_p90_col].astype(float).to_numpy(), a_min=0.0, a_max=None)
        if pred_p90_col in df.columns
        else y_p50 * 1.45
    )

    df["_abs_err"] = np.abs(y_true - y_p50)
    df["_signed_err"] = y_p50 - y_true
    df["_abs_y"] = np.abs(y_true)
    df["_ql_p10"] = _pinball_loss(y_true, y_p10, 0.10)
    df["_ql_p50"] = _pinball_loss(y_true, y_p50, 0.50)
    df["_ql_p90"] = _pinball_loss(y_true, y_p90, 0.90)

    # Error escalado por serie para MASE (usando piso de 0.5 unidades para series Cold Start sin escala histórica)
    scale_denom = np.where(df["naive_scale"].notna() & (df["naive_scale"] > 1e-3), df["naive_scale"], np.maximum(np.mean(np.abs(y_true)), 0.5))
    df["_scaled_err"] = df["_abs_err"] / np.maximum(scale_denom, 0.1)

    records: List[Dict[str, Any]] = []
    valid_group_cols = [c for c in group_cols if c in df.columns]
    if not valid_group_cols:
        df["_all"] = "ALL"
        valid_group_cols = ["_all"]

    for keys, sub in df.groupby(valid_group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_dict = {col: val for col, val in zip(valid_group_cols, keys)}

        denom_y = max(float(sub["_abs_y"].sum()), 1e-6)
        wape = float(sub["_abs_err"].sum() / denom_y)
        bias = float(sub["_signed_err"].sum() / denom_y)
        mase = float(sub["_scaled_err"].mean())
        wql_p10 = float(sub["_ql_p10"].sum() / denom_y)
        wql_p50 = float(sub["_ql_p50"].sum() / denom_y)
        wql_p90 = float(sub["_ql_p90"].sum() / denom_y)
        wql_mean = (wql_p10 + wql_p50 + wql_p90) / 3.0

        n_series = (
            sub[["article_id", "sales_channel_id"]].drop_duplicates().shape[0]
            if {"article_id", "sales_channel_id"}.issubset(sub.columns)
            else len(sub)
        )

        row_dict.update(
            {
                "n_series": int(n_series),
                "total_actual_units": round(float(sub[actual_col].sum()), 2),
                "total_forecast_p50_units": round(float(sub[pred_p50_col].sum()), 2),
                "WAPE": round(wape, 4),
                "MASE": round(mase, 4),
                "WQL_P10": round(wql_p10, 4),
                "WQL_P50": round(wql_p50, 4),
                "WQL_P90": round(wql_p90, 4),
                "WQL_Mean": round(wql_mean, 4),
                "Forecast_Bias": round(bias, 4),
            }
        )
        records.append(row_dict)

    return pd.DataFrame(records)


def compute_forecast_metrics_spark(
    df_eval_spark: Any,
    group_cols: Optional[List[str]] = None,
    actual_col: str = "y_true",
    pred_p50_col: str = "y_hat_p50",
    pred_p10_col: str = "y_hat_p10",
    pred_p90_col: str = "y_hat_p90",
) -> Any:
    """
    Calcula de manera distribuida en PySpark las métricas WAPE, MASE, WQL (P10, P50, P90),
    WQL_Mean y Forecast Bias agrupadas por `group_cols`.
    """
    from pyspark.sql import functions as F

    if group_cols is None:
        group_cols = ["model_name", "demand_regime"]

    y = F.col(actual_col).cast("double")
    p50 = F.greatest(F.col(pred_p50_col).cast("double"), F.lit(0.0))
    p10 = F.greatest(F.col(pred_p10_col).cast("double"), F.lit(0.0))
    p90 = F.greatest(F.col(pred_p90_col).cast("double"), F.lit(0.0))

    err10 = y - p10
    err50 = y - p50
    err90 = y - p90

    ql10 = F.lit(2.0) * F.greatest(F.lit(0.10) * err10, F.lit(-0.90) * err10)
    ql50 = F.lit(2.0) * F.greatest(F.lit(0.50) * err50, F.lit(-0.50) * err50)
    ql90 = F.lit(2.0) * F.greatest(F.lit(0.90) * err90, F.lit(-0.10) * err90)

    naive_scale_col = (
        F.when(F.col("naive_scale").isNotNull() & (F.col("naive_scale") > 0.1), F.col("naive_scale"))
        .otherwise(F.lit(1.0))
        if "naive_scale" in df_eval_spark.columns
        else F.lit(1.0)
    )

    df_work = (
        df_eval_spark.withColumn("_abs_err", F.abs(y - p50))
        .withColumn("_signed_err", p50 - y)
        .withColumn("_abs_y", F.abs(y))
        .withColumn("_ql10", ql10)
        .withColumn("_ql50", ql50)
        .withColumn("_ql90", ql90)
        .withColumn("_scaled_err", F.abs(y - p50) / naive_scale_col)
    )

    denom = F.greatest(F.sum("_abs_y"), F.lit(1e-6))

    agg_df = (
        df_work.groupBy(*group_cols)
        .agg(
            F.countDistinct("article_id", "sales_channel_id").alias("n_series"),
            F.round(F.sum(y), 2).alias("total_actual_units"),
            F.round(F.sum(p50), 2).alias("total_forecast_p50_units"),
            F.round(F.sum("_abs_err") / denom, 4).alias("WAPE"),
            F.round(F.avg("_scaled_err"), 4).alias("MASE"),
            F.round(F.sum("_ql10") / denom, 4).alias("WQL_P10"),
            F.round(F.sum("_ql50") / denom, 4).alias("WQL_P50"),
            F.round(F.sum("_ql90") / denom, 4).alias("WQL_P90"),
            F.round(((F.sum("_ql10") + F.sum("_ql50") + F.sum("_ql90")) / F.lit(3.0)) / denom, 4).alias(
                "WQL_Mean"
            ),
            F.round(F.sum("_signed_err") / denom, 4).alias("Forecast_Bias"),
        )
        .orderBy(*group_cols)
    )
    return agg_df


def simulate_periodic_review_inventory(
    df_forecasts: pd.DataFrame,
    lead_time_weeks: int = 2,
    review_period_weeks: int = 1,
    target_service_level: float = 0.90,
    holding_cost_per_unit_week: float = 0.50,
    stockout_penalty_per_unit: float = 5.00,
    initial_stock_weeks_cover: float = 2.0,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simulación operativa de inventario orientada a decisiones (*Decision-Aware Evaluation*)
    bajo una política de revisión periódica `(R, S)` (*Order-Up-To Level*) durante el
    horizonte de prueba (ej. H = 12 semanas).

    Dinámica semanal por cada SKU-canal (`article_id`, `sales_channel_id`):
    1. Recepción de pedidos en tránsito que cumplen su Lead Time (`Lt` semanas).
    2. Atención de la demanda real observada `y_true[t]`:
       - `sales_satisfied = min(on_hand, y_true[t])`
       - `lost_sales = max(0, y_true[t] - on_hand)` (quiebre de stock sin backorders en retail moda)
       - `on_hand = on_hand - sales_satisfied`
    3. Revisión de inventario cada `R` semanas:
       - El nivel objetivo de reposición `S_t` cubre el periodo de protección `P = Lt + R` semanas.
       - Si el modelo provee cuantiles (`y_hat_p90` y `y_hat_p50`), el stock objetivo para un
         nivel de servicio `alpha` interpola entre la demanda esperada `P50` y el cuantil superior `P90`
         (escalado por la raíz del tiempo de protección):
         `S_t = P * y_hat_p50 + z_ratio * sqrt(P) * max(0, y_hat_p90 - y_hat_p50)`
       - El pedido emitido al proveedor es `order_qty = max(0, S_t - (on_hand + in_transit))`.

    Returns
    -------
    pd.DataFrame
        Resumen consolidado por `group_cols` (ej. `['model_name', 'demand_regime']`) con:
        `['Fill_Rate', 'Lost_Sales_Units', 'Avg_On_Hand_Stock', 'Overstock_Ratio',
          'Holding_Cost_USD', 'Stockout_Penalty_USD', 'Total_Inventory_Cost_USD']`
    """
    from statistics import NormalDist

    if group_cols is None:
        group_cols = ["model_name", "demand_regime"]

    df = df_forecasts.copy()
    sort_cols = [c for c in ["model_name", "article_id", "sales_channel_id", "week_start"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    protection_weeks = float(lead_time_weeks + review_period_weeks)
    # Factor z relativo al cuantil P90 (z_0.90 = 1.28155) usando la librería estándar statistics.NormalDist
    z_90 = 1.2815515655446004
    z_target = float(NormalDist().inv_cdf(float(np.clip(target_service_level, 0.50, 0.995))))
    quantile_scale = z_target / z_90

    series_keys = [c for c in ["model_name", "demand_regime", "article_id", "sales_channel_id"] if c in df.columns]
    sim_records: List[Dict[str, Any]] = []

    for keys, sub in df.groupby(series_keys, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = {k: v for k, v in zip(series_keys, keys)}

        y_true = np.maximum(sub["y_true"].astype(float).to_numpy(), 0.0)
        y_p50 = np.maximum(sub["y_hat_p50"].astype(float).to_numpy(), 0.0)
        y_p90 = (
            np.maximum(sub["y_hat_p90"].astype(float).to_numpy(), y_p50)
            if "y_hat_p90" in sub.columns
            else y_p50 * 1.40
        )

        horizon = len(y_true)
        # Inventario inicial basado en el pronóstico inicial de las primeras semanas
        init_weekly_est = float(np.mean(y_p50[: min(lead_time_weeks, horizon)])) if horizon > 0 else 0.0
        on_hand = max(0.0, initial_stock_weeks_cover * init_weekly_est)

        # Cola FIFO de pedidos en tránsito: longitud igual a lead_time_weeks
        pipeline = [0.0 for _ in range(max(lead_time_weeks, 1))]

        total_demand = 0.0
        total_satisfied = 0.0
        total_lost = 0.0
        on_hand_history: List[float] = []

        for t in range(horizon):
            # 1. Llega el pedido emitido hace `lead_time_weeks`
            arriving = pipeline.pop(0) if lead_time_weeks > 0 else 0.0
            on_hand += arriving

            # 2. Se observa la demanda de la semana t
            d_t = y_true[t]
            satisfied = min(on_hand, d_t)
            lost = max(0.0, d_t - on_hand)
            on_hand -= satisfied

            total_demand += d_t
            total_satisfied += satisfied
            total_lost += lost
            on_hand_history.append(on_hand)

            # 3. Revisión periódica y emisión de orden de reabastecimiento
            if (t % max(review_period_weeks, 1)) == 0:
                # Pronóstico promedio por semana para las próximas semanas
                future_slice_p50 = y_p50[t : min(t + int(protection_weeks), horizon)]
                future_slice_p90 = y_p90[t : min(t + int(protection_weeks), horizon)]
                mean_f_p50 = float(np.mean(future_slice_p50)) if len(future_slice_p50) > 0 else 0.0
                mean_f_p90 = float(np.mean(future_slice_p90)) if len(future_slice_p90) > 0 else 0.0

                spread_90 = max(0.0, mean_f_p90 - mean_f_p50)
                safety_stock = quantile_scale * np.sqrt(protection_weeks) * spread_90
                order_up_to_s = (protection_weeks * mean_f_p50) + safety_stock

                inventory_position = on_hand + sum(pipeline)
                order_qty = max(0.0, order_up_to_s - inventory_position)
                if lead_time_weeks > 0:
                    pipeline.append(order_qty)
                else:
                    on_hand += order_qty
            else:
                if lead_time_weeks > 0:
                    pipeline.append(0.0)

        avg_on_hand = float(np.mean(on_hand_history)) if on_hand_history else 0.0
        holding_cost = float(np.sum(on_hand_history) * holding_cost_per_unit_week)
        stockout_cost = float(total_lost * stockout_penalty_per_unit)

        key_map.update(
            {
                "series_demand": total_demand,
                "series_satisfied": total_satisfied,
                "series_lost": total_lost,
                "series_avg_on_hand": avg_on_hand,
                "series_holding_cost": holding_cost,
                "series_stockout_cost": stockout_cost,
                "series_total_cost": holding_cost + stockout_cost,
            }
        )
        sim_records.append(key_map)

    df_series_sim = pd.DataFrame(sim_records)
    valid_group_cols = [c for c in group_cols if c in df_series_sim.columns]
    if not valid_group_cols:
        df_series_sim["_all"] = "ALL"
        valid_group_cols = ["_all"]

    agg = df_series_sim.groupby(valid_group_cols, as_index=False).agg(
        n_series=("series_demand", "count"),
        total_demand_units=("series_demand", "sum"),
        satisfied_units=("series_satisfied", "sum"),
        Lost_Sales_Units=("series_lost", "sum"),
        Avg_On_Hand_Stock_Per_Series=("series_avg_on_hand", "mean"),
        Total_Avg_On_Hand_Stock=("series_avg_on_hand", "sum"),
        Holding_Cost_USD=("series_holding_cost", "sum"),
        Stockout_Penalty_USD=("series_stockout_cost", "sum"),
        Total_Inventory_Cost_USD=("series_total_cost", "sum"),
    )

    agg["Fill_Rate"] = np.where(
        agg["total_demand_units"] > 1e-6,
        agg["satisfied_units"] / agg["total_demand_units"],
        1.0,
    )
    agg["Overstock_Ratio"] = np.where(
        agg["total_demand_units"] > 1e-6,
        agg["Total_Avg_On_Hand_Stock"] / (agg["total_demand_units"] / max(horizon, 1)),
        0.0,
    )
    agg["target_service_level"] = target_service_level
    return agg.round(4)


def compute_service_efficiency_curve(
    df_forecasts: pd.DataFrame,
    service_levels: Optional[List[float]] = None,
    lead_time_weeks: int = 2,
    review_period_weeks: int = 1,
) -> pd.DataFrame:
    """
    Evalúa la frontera de eficiencia de inventario variando la meta de nivel de servicio
    (ej. `[0.75, 0.85, 0.90, 0.95, 0.98]`) para graficar la curva de compensación
    (Trade-off Curve) entre Fill Rate real alcanzado y Stock Promedio Inmovilizado.
    """
    if service_levels is None:
        service_levels = [0.75, 0.85, 0.90, 0.95, 0.98]

    frames: List[pd.DataFrame] = []
    for sl in service_levels:
        res = simulate_periodic_review_inventory(
            df_forecasts=df_forecasts,
            lead_time_weeks=lead_time_weeks,
            review_period_weeks=review_period_weeks,
            target_service_level=sl,
            group_cols=["model_name"],
        )
        frames.append(res)

    return pd.concat(frames, ignore_index=True)
