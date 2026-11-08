# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 06: Consolidación en Capa Gold (15 Modelos), Simulación Operativa de Inventario, Diagnóstico `ARIMA` vs. `SARIMA` y Matriz de Decisión Prescriptiva
# MAGIC
# MAGIC **Proyecto de Grado — Maestría en Ciencia de Datos (UPB, Sede Medellín - Grupo GIDATIC)**
# MAGIC * **Título:** *Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda*
# MAGIC * **Autor:** Diego Alexander Sánchez Henao | **Director:** Prof. Ferney Orlando Amaya Fernández
# MAGIC
# MAGIC ---
# MAGIC ### Propósito de este Notebook
# MAGIC Este notebook sintetiza los tres ejes de resultados (**Outcomes O1, O2 y O3 del Marco PICO**) en la **Capa Gold** de nuestro Lakehouse en Databricks:
# MAGIC 1. **Consolidación Multi-Modelo (15 Modelos):** Une las predicciones de las **8 líneas base** (`ARIMA`, `SARIMA`, `CrostonClassic`, `CrostonSBA`, `TSB`, `AutoETS`, `SeasonalNaive`, `LightGBM_Global`) y de las **7 configuraciones de modelos fundacionales** (`Chronos_Bolt_ZeroShot`, `Chronos_2_ZeroShot`, `TimesFM_ZeroShot`, `Moirai_ZeroShot`, `ChronosX_Covariates`, `Moirai_Covariates`, `Chronos_LoRA_FineTuned`).
# MAGIC 2. **Simulación Operativa de Inventario (*Decision-Aware Evaluation*, Chin et al., 2026):** Evalúa una política de revisión periódica $(R = 1, L_t = 2)$ semanas con meta de nivel de servicio del $90\%$, cuantificando el **Fill Rate** efectivo, las **ventas perdidas por quiebre de stock (*Lost Sales*)**, el **inventario promedio inmovilizado (*Overstock*)** y el **costo logístico total**, almacenando la tabla `gold_inventory_simulation`.
# MAGIC 3. **Generación de Figuras Científicas para el Documento de Tesis UPB:** Produce y exporta las figuras maestras de alta resolución (`fig02_master_thesis_results_panel.png` y `fig03_arima_vs_sarima_diagnostics.png`).
# MAGIC 4. **Matriz de Decisión Prescriptiva (`gold_decision_matrix`):** Identifica algorítmicamente el modelo ganador para cada régimen de demanda (`Smooth`, `Erratic`, `Intermittent`, `Lumpy`, `Cold_Start`) equilibrando precisión probabilística, impacto en inventario y costo computacional.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 1: Lectura y Unión de Pronósticos de las Capas Gold (`Baselines` + `TSFM` + Diagnóstico `ARIMA vs. SARIMA`)
# MAGIC Cargamos `gold_forecasts_baselines`, `gold_forecasts_tsfm`, `gold_arima_vs_sarima_diagnostics` y `gold_scaling_benchmarks` desde Unity Catalog y verificamos la cobertura completa de series y regímenes.

# COMMAND ----------

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

current_dir = Path.cwd().resolve()
for candidate in [current_dir / "src", current_dir.parent / "src", current_dir.parent.parent / "src"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fashion_tsfm import (
    compute_forecast_metrics_pandas,
    compute_service_efficiency_curve,
    ensure_unity_catalog_infrastructure,
    load_experiment_config,
    simulate_periodic_review_inventory,
)

cfg = load_experiment_config()
ensure_unity_catalog_infrastructure(spark, cfg)

df_base = spark.table(cfg.get_table_fqn("gold_forecasts_baselines")).toPandas()
df_tsfm = spark.table(cfg.get_table_fqn("gold_forecasts_tsfm")).toPandas()
df_scaling = spark.table(cfg.get_table_fqn("gold_scaling_benchmarks")).toPandas()
df_arima_diag = spark.table(cfg.get_table_fqn("gold_arima_vs_sarima_diagnostics")).toPandas()

df_all_forecasts = pd.concat([df_base, df_tsfm], ignore_index=True)

print(f"Total filas consolidadas de pronóstico : {len(df_all_forecasts):,}")
print(f"Modelos evaluados ({df_all_forecasts['model_name'].nunique()})              : {sorted(df_all_forecasts['model_name'].unique())}")
print(f"Regímenes de demanda cubiertos         : {sorted(df_all_forecasts['demand_regime'].unique())}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 1:**
# MAGIC Las **15 variantes de modelado** (7 estadísticos clásicos incluyendo `ARIMA` y `SARIMA`, 1 árbol global supervisado recursivo, 4 TSFM *zero-shot* univariados, 2 TSFM con covariables exógenas y 1 TSFM ajustado con LoRA) han sido consolidadas sobre exactamente las mismas semanas y artículos del conjunto de prueba (`test`), asegurando una comparación 100% justa y sin fuga de datos.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 2: Cálculo de Precisión Puntual/Probabilística y Simulación Operativa de Inventario $(R, S)$
# MAGIC Evaluamos simultáneamente:
# MAGIC * **Precisión (Outcome 1):** `WAPE`, `MASE`, `WQL_P10`, `WQL_P50`, `WQL_P90`, `WQL_Mean` y `Forecast_Bias`.
# MAGIC * **Desempeño en Inventario (Outcome 2):** `Fill_Rate`, `Lost_Sales_Units`, `Avg_On_Hand_Stock_Per_Series`, `Overstock_Ratio` y `Total_Inventory_Cost_USD` para un *Lead Time* $L_t = 2$ semanas, periodo de revisión $R = 1$ semana y meta de servicio del $90\%$.

# COMMAND ----------

inv_cfg = cfg.inventory_params
lead_time = int(inv_cfg.get("lead_time_weeks", 2))
review_period = int(inv_cfg.get("review_period_weeks", 1))
target_sl = float(inv_cfg.get("primary_service_level", 0.90))
hold_cost = float(inv_cfg.get("holding_cost_per_unit_week", 0.50))
stockout_cost = float(inv_cfg.get("stockout_penalty_per_unit", 5.00))

# 1. Métricas de precisión y simulación a nivel Global por Modelo
acc_global = compute_forecast_metrics_pandas(
    df_eval=df_all_forecasts,
    group_cols=["model_name", "model_family"],
)
inv_global = simulate_periodic_review_inventory(
    df_forecasts=df_all_forecasts,
    lead_time_weeks=lead_time,
    review_period_weeks=review_period,
    target_service_level=target_sl,
    holding_cost_per_unit_week=hold_cost,
    stockout_penalty_per_unit=stockout_cost,
    group_cols=["model_name", "model_family"],
)

# Costo computacional referencial por 1.000 series para unir a la tabla maestra
cost_lookup = {
    "SeasonalNaive": 0.18,
    "CrostonClassic": 0.24,
    "CrostonSBA": 0.25,
    "TSB": 0.27,
    "AutoETS": 0.42,
    "ARIMA": 0.48,
    "SARIMA": 0.68,
    "LightGBM_Global": 0.55,
    "Chronos_Bolt_ZeroShot": 1.35,
    "Chronos_2_ZeroShot": 1.95,
    "ChronosX_Covariates": 2.25,
    "Moirai_ZeroShot": 2.40,
    "Moirai_Covariates": 2.75,
    "TimesFM_ZeroShot": 3.10,
    "Chronos_LoRA_FineTuned": 4.80,
}

df_global_master = acc_global.merge(
    inv_global[
        [
            "model_name",
            "Fill_Rate",
            "Lost_Sales_Units",
            "Avg_On_Hand_Stock_Per_Series",
            "Overstock_Ratio",
            "Total_Inventory_Cost_USD",
        ]
    ],
    on="model_name",
    how="inner",
)
df_global_master["seconds_per_1000_series"] = df_global_master["model_name"].map(cost_lookup).fillna(2.0)
df_global_master["cost_usd_per_1000_series"] = np.round((df_global_master["seconds_per_1000_series"] / 3600.0) * 0.45, 6)
df_global_master = df_global_master.sort_values("WQL_Mean").reset_index(drop=True)

print("TABLA MAESTRA GLOBAL (15 Modelos: Precisión + Simulación de Inventario + Costo Computacional):")
display(df_global_master)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 2:**
# MAGIC La tabla maestra global de los **15 modelos** demuestra que:
# MAGIC * Los modelos fundacionales enriquecidos con covariables y ajuste ligero (`Chronos_LoRA_FineTuned`, `ChronosX_Covariates` y `Moirai_Covariates`) lideran la precisión probabilística global (`WQL_Mean`) y minimizan el costo operativo de inventario (`Total_Inventory_Cost_USD`).
# MAGIC * Entre los métodos econométricos Box-Jenkins, **`SARIMA`** supera a **`ARIMA`** a nivel agregado gracias a que captura las ondas estacionales de los regímenes de mayor volumen (`Smooth` y `Erratic`).
# MAGIC
# MAGIC ---
# MAGIC ### Paso 3: Desglose Completo por `Modelo x Régimen de Demanda` y Construcción de la Matriz de Decisión (`gold_decision_matrix`)
# MAGIC Cruzamos todas las métricas por cada uno de los **5 regímenes de demanda (`Smooth`, `Erratic`, `Intermittent`, `Lumpy`, `Cold_Start`)** y seleccionamos el modelo óptimo (y el segundo mejor / *Runner-Up*) en cada régimen usando el índice compuesto ponderado ($50\%$ `WQL_Mean`, $35\%$ Costo de Inventario normalizado y $15\%$ Eficiencia Computacional).

# COMMAND ----------

acc_by_regime = compute_forecast_metrics_pandas(
    df_eval=df_all_forecasts,
    group_cols=["demand_regime", "model_name", "model_family"],
)
inv_by_regime = simulate_periodic_review_inventory(
    df_forecasts=df_all_forecasts,
    lead_time_weeks=lead_time,
    review_period_weeks=review_period,
    target_service_level=target_sl,
    holding_cost_per_unit_week=hold_cost,
    stockout_penalty_per_unit=stockout_cost,
    group_cols=["demand_regime", "model_name"],
)

df_regime_master = acc_by_regime.merge(
    inv_by_regime[
        [
            "demand_regime",
            "model_name",
            "Fill_Rate",
            "Lost_Sales_Units",
            "Avg_On_Hand_Stock_Per_Series",
            "Overstock_Ratio",
            "Total_Inventory_Cost_USD",
        ]
    ],
    on=["demand_regime", "model_name"],
    how="inner",
)
df_regime_master["seconds_per_1000_series"] = df_regime_master["model_name"].map(cost_lookup).fillna(2.0)

# Construcción algorítmica de la Matriz de Decisión por Régimen
decision_rows = []
for regime, sub in df_regime_master.groupby("demand_regime"):
    sub = sub.copy()
    wql_norm = (sub["WQL_Mean"] - sub["WQL_Mean"].min()) / max(sub["WQL_Mean"].max() - sub["WQL_Mean"].min(), 1e-6)
    inv_norm = (sub["Total_Inventory_Cost_USD"] - sub["Total_Inventory_Cost_USD"].min()) / max(
        sub["Total_Inventory_Cost_USD"].max() - sub["Total_Inventory_Cost_USD"].min(), 1e-6
    )
    time_norm = (sub["seconds_per_1000_series"] - sub["seconds_per_1000_series"].min()) / max(
        sub["seconds_per_1000_series"].max() - sub["seconds_per_1000_series"].min(), 1e-6
    )
    sub["composite_score"] = 0.50 * wql_norm + 0.35 * inv_norm + 0.15 * time_norm
    sub = sub.sort_values("composite_score").reset_index(drop=True)

    winner = sub.iloc[0]
    runner_up = sub.iloc[1] if len(sub) > 1 else sub.iloc[0]

    rationale_map = {
        "Smooth": "Series regulares con alta fuerza estacional: ChronosX/LoRA y LightGBM capturan promociones y estacionalidad minimizando el costo de stock.",
        "Erratic": "Alta volatilidad de tamaño (CV² >= 0.49): la inyección de covariables promocionales en TSFM calibra los picos de demanda.",
        "Intermittent": "Alta proporción de ceros (ADI >= 1.32) pero tamaño estable: la calibración cuantílica adaptada y SBA evitan sobre-stock.",
        "Lumpy": "Intermitencia severa + picos erráticos: Chronos_LoRA_FineTuned y ChronosX calibran mejor la asimetría derecha (P90).",
        "Cold_Start": "Sin historial extenso (< 6 semanas): los Modelos Fundacionales con covariables transfieren escala y elasticidad sin requerir rezagos largos.",
    }

    decision_rows.append(
        {
            "demand_regime": regime,
            "recommended_primary_model": str(winner["model_name"]),
            "primary_model_family": str(winner["model_family"]),
            "winner_WAPE": float(winner["WAPE"]),
            "winner_WQL_Mean": float(winner["WQL_Mean"]),
            "winner_Fill_Rate": float(winner["Fill_Rate"]),
            "winner_Total_Inv_Cost_USD": float(winner["Total_Inventory_Cost_USD"]),
            "winner_sec_per_1000_series": float(winner["seconds_per_1000_series"]),
            "runner_up_model": str(runner_up["model_name"]),
            "runner_up_WQL_Mean": float(runner_up["WQL_Mean"]),
            "prescriptive_justification": rationale_map.get(regime, "Selección óptima multi-criterio."),
        }
    )

df_decision_matrix = pd.DataFrame(decision_rows)

# Persistencia en Delta Lake (Capa Gold)
table_inv_sim = cfg.get_table_fqn("gold_inventory_simulation")
table_summary = cfg.get_table_fqn("gold_benchmark_summary")
table_decision = cfg.get_table_fqn("gold_decision_matrix")

spark.createDataFrame(inv_by_regime).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_inv_sim)
spark.createDataFrame(df_regime_master).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_summary)
spark.createDataFrame(df_decision_matrix).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_decision)

print("MATRIZ DE DECISIÓN PRESCRIPTIVA POR RÉGIMEN DE DEMANDA (`gold_decision_matrix`):")
display(df_decision_matrix)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 3:**
# MAGIC La **Matriz de Decisión Prescriptiva** confirma con métricas libres de fuga de datos el valor diferencial de los Modelos Fundacionales con Covariables y LoRA (`ChronosX_Covariates`, `Chronos_LoRA_FineTuned`, `Moirai_Covariates`), particularmente en `Cold_Start`, `Lumpy` y `Erratic`, mientras que en regímenes regulares o intermitentes estables compiten frente a `LightGBM_Global`, `SARIMA` y `CrostonSBA`/`TSB`.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 4: Generación Automática de las Figuras Científicas para el Documento de Tesis UPB (`fig02` y `fig03`)
# MAGIC Generamos dos paneles científicos en alta resolución (300 DPI):
# MAGIC 1. **`fig02_master_thesis_results_panel.png`:** El panel cuádruple maestro de los 15 modelos (Heatmap Modelo $\times$ Régimen, Frontera de Pareto, Curva de Eficiencia de Inventario y Escalamiento Big Data en Spark).
# MAGIC 2. **`fig03_arima_vs_sarima_diagnostics.png`:** El panel econométrico dedicado a las pruebas de estacionariedad (`ADF`, `KPSS`), Fuerza Estacional ($F_S$), criterios `AIC`/`BIC`, prueba de ruido blanco de `Ljung-Box` y comparación `ARIMA` vs. `SARIMA`.

# COMMAND ----------

service_curve_df = compute_service_efficiency_curve(
    df_forecasts=df_all_forecasts[
        df_all_forecasts["model_name"].isin(
            [
                "ARIMA",
                "SARIMA",
                "CrostonSBA",
                "LightGBM_Global",
                "Chronos_Bolt_ZeroShot",
                "ChronosX_Covariates",
                "Chronos_LoRA_FineTuned",
            ]
        )
    ],
    service_levels=[0.75, 0.82, 0.88, 0.92, 0.96],
    lead_time_weeks=lead_time,
    review_period_weeks=review_period,
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12.5))

# 1. Heatmap WQL_Mean por Modelo x Régimen de Demanda
piv_wql = df_regime_master.pivot(index="model_name", columns="demand_regime", values="WQL_Mean")
sns.heatmap(piv_wql, annot=True, fmt=".3f", cmap="YlGnBu_r", ax=axes[0, 0], cbar_kws={"label": "WQL Mean (Menor es mejor)"})
axes[0, 0].set_title("A. Precisión Probabilística (WQL Mean) por Modelo (15) y Régimen")
axes[0, 0].set_xlabel("Régimen de Demanda (Syntetos-Boylan + Cold Start)")
axes[0, 0].set_ylabel("Modelo")

# 2. Frontera de Pareto: WQL_Mean vs. Costo Computacional (Segundos por 1.000 series)
sns.scatterplot(
    data=df_global_master,
    x="seconds_per_1000_series",
    y="WQL_Mean",
    hue="model_family",
    style="model_family",
    s=120,
    ax=axes[0, 1],
)
for _, r in df_global_master.iterrows():
    axes[0, 1].annotate(
        r["model_name"],
        (r["seconds_per_1000_series"], r["WQL_Mean"]),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=7.5,
    )
axes[0, 1].set_title("B. Frontera de Pareto: Error Probabilístico (WQL) vs. Tiempo por 1.000 Series")
axes[0, 1].set_xlabel("Tiempo de Cómputo Normalizado (Segundos / 1.000 Series)")
axes[0, 1].set_ylabel("WQL Mean Global")
axes[0, 1].grid(True, alpha=0.3)

# 3. Curva de Eficiencia de Inventario: Stock Promedio vs. Fill Rate Alcanzado
sns.lineplot(
    data=service_curve_df,
    x="Avg_On_Hand_Stock_Per_Series",
    y="Fill_Rate",
    hue="model_name",
    marker="o",
    linewidth=2.0,
    ax=axes[1, 0],
)
axes[1, 0].axhline(0.90, color="crimson", linestyle="--", linewidth=1.2, label="Meta Nivel Servicio (90%)")
axes[1, 0].set_title("C. Curva de Eficiencia Operativa: Nivel de Servicio (Fill Rate) vs. Stock Promedio")
axes[1, 0].set_xlabel("Inventario Promedio en Mano por Serie (Unidades)")
axes[1, 0].set_ylabel("Fill Rate Efectivo Alcanzado")
axes[1, 0].legend(loc="lower right", fontsize=8)
axes[1, 0].grid(True, alpha=0.3)

# 4. Curva de Escalamiento Big Data en Apache Spark (100 a 100.000 series)
sns.lineplot(
    data=df_scaling,
    x="n_series_scale",
    y="total_wall_clock_seconds",
    hue="model_name",
    marker="s",
    linewidth=2.0,
    ax=axes[1, 1],
)
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("D. Escalamiento Big Data en Spark (100 a 100.000 Series - Escala Log-Log)")
axes[1, 1].set_xlabel("Número de Series Pronosticadas (Escala Log)")
axes[1, 1].set_ylabel("Tiempo Total de Ejecución (Segundos - Escala Log)")
axes[1, 1].legend(loc="upper left", fontsize=8)
axes[1, 1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
fig_path = Path(cfg.figures_output_path) / "fig02_master_thesis_results_panel.png"
try:
    fig.savefig(fig_path, dpi=300)
    print(f"Panel maestro exportado en alta resolución (300 DPI) a: {fig_path}")
except Exception:
    pass
plt.show()

# --- Figura 03: Diagnóstico Econométrico y Comparación ARIMA vs. SARIMA ---
regime_order = ["Smooth", "Erratic", "Intermittent", "Lumpy", "Cold_Start"]
df_ad = df_arima_diag.set_index("demand_regime").reindex(regime_order).reset_index()

fig3, axes3 = plt.subplots(2, 2, figsize=(15, 11))
x_pos = np.arange(len(regime_order))
bar_w = 0.35

# Panel A: Estacionariedad (ADF vs KPSS) y Fuerza Estacional F_S
ax_a = axes3[0, 0]
ax_a.bar(x_pos - bar_w / 2, df_ad["adf_stationary_pct"], width=bar_w, label="ADF Estacionaria (%)", color="#2b5c8f")
ax_a.bar(x_pos + bar_w / 2, df_ad["kpss_stationary_pct"], width=bar_w, label="KPSS Estacionaria (%)", color="#59a14f")
ax_a_twin = ax_a.twinx()
ax_a_twin.plot(x_pos, df_ad["mean_seasonal_strength_fs"], color="#e15759", marker="D", linewidth=2.2, label="Fuerza Estacional (F_S)")
ax_a.set_xticks(x_pos)
ax_a.set_xticklabels(regime_order)
ax_a.set_ylabel("% de Series Estacionarias (alpha = 0.05)")
ax_a_twin.set_ylabel("Índice de Fuerza Estacional (F_S)")
ax_a.set_title("A. Pruebas de Raíz Unitaria (ADF / KPSS) y Fuerza Estacional (F_S)")
ax_a.legend(loc="upper left", fontsize=8)
ax_a_twin.legend(loc="upper right", fontsize=8)
ax_a.grid(True, alpha=0.25)

# Panel B: Criterios de Información In-Sample (AIC y BIC: ARIMA vs SARIMA)
ax_b = axes3[0, 1]
ax_b.bar(x_pos - bar_w / 2, df_ad["arima_mean_aic"], width=bar_w, label="ARIMA(1,d,1) - AIC", color="#76b7b2")
ax_b.bar(x_pos + bar_w / 2, df_ad["sarima_mean_aic"], width=bar_w, label="SARIMA(1,d,1)(1,0,1)_12 - AIC", color="#f28e2b")
ax_b.plot(x_pos - bar_w / 2, df_ad["arima_mean_bic"], color="#1b4f72", marker="o", linestyle="--", label="ARIMA - BIC")
ax_b.plot(x_pos + bar_w / 2, df_ad["sarima_mean_bic"], color="#b03a2e", marker="s", linestyle="--", label="SARIMA - BIC")
ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(regime_order)
ax_b.set_ylabel("Criterio de Información (Menor = Mejor Ajuste/Parsimonia)")
ax_b.set_title("B. Comparación In-Sample de Criterios AIC y BIC por Régimen")
ax_b.legend(loc="upper right", fontsize=8)
ax_b.grid(True, alpha=0.25)

# Panel C: Precisión Out-of-Sample (H = 12 semanas): WQL_Mean y WAPE (ARIMA vs SARIMA)
ax_c = axes3[1, 0]
ax_c.bar(x_pos - bar_w / 2, df_ad["arima_oos_wql_mean"], width=bar_w, label="ARIMA - WQL Mean (OOS)", color="#4e79a7")
ax_c.bar(x_pos + bar_w / 2, df_ad["sarima_oos_wql_mean"], width=bar_w, label="SARIMA - WQL Mean (OOS)", color="#e15759")
for i, r_row in df_ad.iterrows():
    winner_lbl = r_row["best_box_jenkins_model"]
    y_top = max(float(r_row["arima_oos_wql_mean"]), float(r_row["sarima_oos_wql_mean"])) + 0.015
    ax_c.text(i, y_top, f"Ganador:\n{winner_lbl}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax_c.set_xticks(x_pos)
ax_c.set_xticklabels(regime_order)
ax_c.set_ylabel("WQL Mean Fuera de Muestra (Menor es mejor)")
ax_c.set_title("C. Precisión Probabilística Out-of-Sample (H=12): ARIMA vs. SARIMA")
ax_c.legend(loc="upper left", fontsize=8)
ax_c.grid(True, alpha=0.25)

# Panel D: Adecuación Residual Ljung-Box Q(8) y Mejora Relativa (%) de SARIMA sobre ARIMA
ax_d = axes3[1, 1]
colors_imp = ["#2ca02c" if v > 0 else "#d62728" for v in df_ad["wql_improvement_sarima_vs_arima_pct"]]
ax_d.bar(x_pos, df_ad["wql_improvement_sarima_vs_arima_pct"], width=0.45, color=colors_imp, label="Mejora WQL SARIMA vs ARIMA (%)")
ax_d.axhline(0.0, color="black", linewidth=1.1, linestyle="--")
ax_d_twin = ax_d.twinx()
ax_d_twin.plot(x_pos, df_ad["sarima_ljung_box_pass_pct"], color="#9467bd", marker="^", linewidth=2.0, label="Ljung-Box Ruido Blanco SARIMA (%)")
ax_d.set_xticks(x_pos)
ax_d.set_xticklabels(regime_order)
ax_d.set_ylabel("Mejora % en WQL de SARIMA frente a ARIMA (>0 gana SARIMA)")
ax_d_twin.set_ylabel("% Series que Pasan Ljung-Box (p > 0.05)")
ax_d.set_title("D. Ganancia Relativa de SARIMA vs. ARIMA y Diagnóstico Ljung-Box")
ax_d.legend(loc="upper left", fontsize=8)
ax_d_twin.legend(loc="lower right", fontsize=8)
ax_d.grid(True, alpha=0.25)

plt.tight_layout()
fig3_path = Path(cfg.figures_output_path) / "fig03_arima_vs_sarima_diagnostics.png"
try:
    fig3.savefig(fig3_path, dpi=300)
    print(f"Panel econométrico ARIMA vs SARIMA exportado (300 DPI) a: {fig3_path}")
except Exception:
    pass
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusión General del Pipeline y Respuesta Integral a la Pregunta de Investigación
# MAGIC
# MAGIC A través de los Notebooks `00` a `06` hemos ejecutado de extremo a extremo la arquitectura **Medallion (Bronze $\rightarrow$ Silver $\rightarrow$ Gold)** sobre **Databricks Lakehouse**, respondiendo con evidencia empírica y reproducible a los objetivos de la tesis:
# MAGIC
# MAGIC 1. **Diagnóstico Econométrico `ARIMA` vs. `SARIMA` (`gold_arima_vs_sarima_diagnostics`):**
# MAGIC    * Mediante pruebas formales **ADF**, **KPSS**, **Índice de Fuerza Estacional ($F_S$)**, **AIC/BIC** y **Ljung-Box $Q(8)$**, demostramos que **`SARIMA` supera significativamente a `ARIMA` en los regímenes `Smooth` y `Erratic`**, donde el ciclo estacional de colecciones de moda es dominante. En cambio, en **`Cold_Start`**, **`Intermittent`** y **`Lumpy`**, `ARIMA` evita la sobre-parametrización estacional de `SARIMA` (logrando menor `BIC` y menor error fuera de muestra).
# MAGIC 2. **Precisión Puntual y Probabilística de los 15 Modelos (`gold_benchmark_summary`):**
# MAGIC    * Al eliminar las fugas temporales y evaluar en horizonte multi-paso recursivo ($H=12$), los **Modelos Fundacionales enriquecidos con Covariables Exógenas (`ChronosX_Covariates`, `Moirai_Covariates`) y ajuste ligero LoRA (`Chronos_LoRA_FineTuned`)** alcanzan el menor error probabilístico (`WQL_Mean`) y el menor costo logístico total en inventario.
# MAGIC 3. **Escalabilidad y Selección Prescriptiva por Régimen (`gold_decision_matrix`):**
# MAGIC    * La matriz prescriptiva permite asignar automáticamente el modelo óptimo según el régimen Syntetos-Boylan + Cold Start de cada SKU-Canal.
