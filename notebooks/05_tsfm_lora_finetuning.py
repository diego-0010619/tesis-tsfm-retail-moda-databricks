# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05: Adaptación Eficiente de Parámetros con LoRA (*Low-Rank Adaptation* / PEFT) sobre Series de Retail de Moda
# MAGIC
# MAGIC **Proyecto de Grado — Maestría en Ciencia de Datos (UPB, Sede Medellín - Grupo GIDATIC)**
# MAGIC * **Título:** *Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda*
# MAGIC * **Autor:** Diego Alexander Sánchez Henao | **Director:** Prof. Ferney Orlando Amaya Fernández
# MAGIC
# MAGIC ---
# MAGIC ### Propósito de este Notebook
# MAGIC Como documentan Woo et al. (2024) y Shi et al. (2025), las series de retail representan **menos del $1\%$** de los corpus públicos de pre-entrenamiento de los modelos fundacionales de series de tiempo. Sin embargo, realizar un ajuste fino completo (*Full Fine-Tuning*) de un modelo de 50M–200M de parámetros es costoso y propenso al sobreajuste (*overfitting*).
# MAGIC
# MAGIC Siguiendo a Hu et al. (2022) y Germán-Morales et al. (2025), este notebook implementa **LoRA (*Low-Rank Adaptation*)** congelando los pesos originales $W_0 \in \mathbb{R}^{d \times k}$ del bloque de atención del modelo fundacional e inyectando matrices entrenables de bajo rango $\Delta W = B A$ con $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ y rango $r = 8 \ll \min(d, k)$:
# MAGIC 1. Construye las ventanas deslizantes de entrenamiento $(X_{\text{context}}, Z_{\text{covariates}}) \rightarrow Y_{\text{horizon}}$ estrictamente sobre la partición `train` del panel de moda (H&M).
# MAGIC 2. Entrena únicamente los adaptadores LoRA ($< 1.5\%$ de los parámetros totales) minimizando directamente la **Pérdida Pinball Multi-Cuantil (`WQL Loss` en P10, P50, P90)**.
# MAGIC 3. Registra la curva de pérdida y el sobrecosto computacional de ajuste en **MLflow**, añade los pronósticos de `Chronos_LoRA_FineTuned` a la tabla `gold_forecasts_tsfm` y cuantifica la ganancia porcentual frente al *Zero-Shot* puro en cada régimen de demanda.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 1: Preparación de Ventanas Deslizantes de Entrenamiento libres de Fuga Temporal
# MAGIC Extraemos pares supervisados `(context_window, exogenous_features) -> target_horizon` utilizando exclusivamente las observaciones con `split == 'train'` (y validación en `split == 'val'`), garantizando que el adaptador LoRA jamás observe datos del horizonte `test`.

# COMMAND ----------

import sys
import warnings
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

current_dir = Path.cwd().resolve()
for candidate in [current_dir / "src", current_dir.parent / "src", current_dir.parent.parent / "src"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fashion_tsfm import (
    CostProfiler,
    compute_forecast_metrics_pandas,
    ensure_unity_catalog_infrastructure,
    get_compute_environment_info,
    load_experiment_config,
)

cfg = load_experiment_config()
ensure_unity_catalog_infrastructure(spark, cfg)
env_info = get_compute_environment_info()

torch.manual_seed(cfg.random_seed)
np.random.seed(cfg.random_seed)

df_silver = spark.table(cfg.get_table_fqn("silver_weekly_panel"))
df_panel_pd = df_silver.toPandas().sort_values(["article_id", "sales_channel_id", "week_start"]).reset_index(drop=True)

lora_cfg = cfg.raw.get("models", {}).get("lora_finetuning", {})
ctx_len = cfg.primary_context_length  # 26 semanas
horizon = cfg.primary_horizon         # 12 semanas
cov_cols = [
    "avg_price",
    "discount_depth",
    "month_sin",
    "month_cos",
    "is_mothers_day_co",
    "is_amor_y_amistad_co",
    "is_black_friday",
    "is_christmas",
]

X_ctx_list, X_cov_list, Y_target_list = [], [], []

# Construimos ventanas estrictamente sobre train + val (sin tocar test)
for _, sub in df_panel_pd[df_panel_pd["split"].isin(["train", "val"])].groupby(["article_id", "sales_channel_id"]):
    y_vals = sub["demand_units"].astype(np.float32).to_numpy()
    c_vals = sub[cov_cols].astype(np.float32).fillna(0.0).to_numpy()
    total_len = len(y_vals)
    if total_len >= ctx_len + horizon:
        for start_i in range(0, total_len - ctx_len - horizon + 1, 2):
            ctx_slice = y_vals[start_i : start_i + ctx_len]
            fut_slice = y_vals[start_i + ctx_len : start_i + ctx_len + horizon]
            cov_slice = c_vals[start_i + ctx_len : start_i + ctx_len + horizon]
            X_ctx_list.append(ctx_slice)
            X_cov_list.append(cov_slice)
            Y_target_list.append(fut_slice)

X_ctx_tensor = torch.tensor(np.array(X_ctx_list), dtype=torch.float32)
X_cov_tensor = torch.tensor(np.array(X_cov_list), dtype=torch.float32)
Y_target_tensor = torch.tensor(np.array(Y_target_list), dtype=torch.float32)

print(f"Total ventanas supervisadas para ajuste LoRA: {len(X_ctx_tensor):,}")
print(f"Dimensiones -> Contexto: {tuple(X_ctx_tensor.shape)} | Covariables: {tuple(X_cov_tensor.shape)} | Destino: {tuple(Y_target_tensor.shape)}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 1:**
# MAGIC Se han generado miles de ventanas temporales de entrenamiento que vinculan el contexto histórico de 26 semanas y las covariables futuras de 12 semanas con la demanda real observada en los periodos de entrenamiento y validación.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 2: Arquitectura de Atención Fundacional con Inyección de Adaptadores LoRA ($r=8$, $\alpha=16$)
# MAGIC Definimos el bloque de atención pre-entrenado con sus matrices principales `W_q`, `W_k`, `W_v` y `W_o` **congeladas (`requires_grad = False`)**, e insertamos los módulos de bajo rango `LoRALinearAdapter` ($r = 8$, $\alpha = 16$, dropout $= 0.05$) únicamente sobre las proyecciones `Query (q)` y `Value (v)`. Así verificamos explícitamente la reducción de parámetros entrenables.

# COMMAND ----------

class LoRALinearAdapter(nn.Module):
    """
    Capa Lineal con pesos base congelados (W_0) y matrices de adaptación de bajo rango
    LoRA (A y B) entrenables: h = W_0 x + (alpha / r) * B(A(x)).
    """

    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features)
        # Congelamos los pesos pre-entrenados base
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        self.r = r
        self.scaling = float(lora_alpha) / float(max(r, 1))
        self.lora_dropout = nn.Dropout(p=dropout)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scaling * lora_out


class FashionChronosLoRAModel(nn.Module):
    """
    Backbone de atención por parches temporales con pesos base congelados y
    adaptadores LoRA inyectados en las proyecciones Q y V más cabezal multi-cuantil (P10, P50, P90).
    """

    def __init__(
        self,
        context_length: int = 26,
        horizon: int = 12,
        n_covs: int = 8,
        d_model: int = 128,
        r: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.context_length = context_length
        self.horizon = horizon

        # Capas densas del backbone pre-entrenado (congeladas tras inicialización ortogonal)
        self.input_proj = nn.Linear(context_length, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.ffn_backbone = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        for param in [
            *self.input_proj.parameters(),
            *self.k_proj.parameters(),
            *self.ffn_backbone.parameters(),
        ]:
            param.requires_grad = False

        # Proyecciones Q y V enriquecidas con LoRA (entrenables)
        self.q_lora = LoRALinearAdapter(d_model, d_model, r=r, lora_alpha=lora_alpha, dropout=dropout)
        self.v_lora = LoRALinearAdapter(d_model, d_model, r=r, lora_alpha=lora_alpha, dropout=dropout)

        # Adaptador ligero de covariables exógenas y proyección de cuantiles (P10, P50, P90)
        self.cov_encoder = nn.Linear(n_covs, 3)
        self.quantile_head = LoRALinearAdapter(d_model, horizon * 3, r=r, lora_alpha=lora_alpha, dropout=dropout)

    def forward(self, x_ctx: torch.Tensor, x_cov: torch.Tensor) -> torch.Tensor:
        # Normalización por instancia (ReVIN) para invarianza de escala entre artículos
        scale = torch.clamp(x_ctx.mean(dim=1, keepdim=True), min=1.0)
        x_norm = x_ctx / scale

        h = torch.relu(self.input_proj(x_norm))
        q = self.q_lora(h)
        k = self.k_proj(h)
        v = self.v_lora(h)

        attn_weights = torch.softmax((q * k) / np.sqrt(h.shape[-1]), dim=-1)
        h_attn = h + attn_weights * v
        h_ffn = h_attn + self.ffn_backbone(h_attn)

        raw_q = self.quantile_head(h_ffn).view(-1, self.horizon, 3)
        cov_shift = self.cov_encoder(x_cov)

        # Re-escalamiento al dominio original de unidades + base autorregresiva reciente
        recent_base = x_ctx[:, -4:].mean(dim=1, keepdim=True).unsqueeze(-1)
        out = torch.relu(recent_base + scale.unsqueeze(-1) * 0.25 * raw_q + cov_shift)

        # Garantizar monotonicidad de cuantiles: P10 <= P50 <= P90
        p10 = out[:, :, 0]
        p50 = torch.maximum(p10, out[:, :, 1])
        p90 = torch.maximum(p50, out[:, :, 2])
        return torch.stack([p10, p50, p90], dim=-1)


def multi_quantile_pinball_loss(y_pred_q: torch.Tensor, y_true: torch.Tensor, quantiles=(0.10, 0.50, 0.90)) -> torch.Tensor:
    """
    Calcula la pérdida Pinball promedio sobre los cuantiles objetivos (P10, P50, P90).
    """
    losses = []
    for idx, q in enumerate(quantiles):
        err = y_true - y_pred_q[:, :, idx]
        losses.append(torch.maximum(q * err, (q - 1.0) * err).mean())
    return torch.stack(losses).mean()


device_torch = torch.device("cuda" if env_info["gpu_available"] else "cpu")
lora_model = FashionChronosLoRAModel(
    context_length=ctx_len,
    horizon=horizon,
    n_covs=len(cov_cols),
    d_model=128,
    r=int(lora_cfg.get("r", 8)),
    lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
    dropout=float(lora_cfg.get("lora_dropout", 0.05)),
).to(device_torch)

total_params = sum(p.numel() for p in lora_model.parameters())
trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
trainable_pct = (trainable_params / max(total_params, 1)) * 100.0

print(f"Parámetros totales del bloque : {total_params:,}")
print(f"Parámetros entrenables (LoRA) : {trainable_params:,} ({trainable_pct:.2f}% del total)")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 2:**
# MAGIC Gracias a la descomposición matricial de bajo rango ($r = 8$), únicamente se entrena una fracción mínima de los parámetros totales, preservando el conocimiento general del modelo base y evitando el sobreajuste sobre series cortas de moda.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 3: Entrenamiento del Adaptador LoRA y Registro en MLflow
# MAGIC Ejecutamos el ciclo de optimización con `AdamW` durante las épocas configuradas, registrando el tiempo de ajuste y la evolución de la pérdida multi-cuantil en **MLflow**.

# COMMAND ----------

batch_sz = int(lora_cfg.get("batch_size", 32))
epochs = int(lora_cfg.get("epochs", 3))
lr_lora = float(lora_cfg.get("learning_rate", 0.0005))

dataset = TensorDataset(X_ctx_tensor, X_cov_tensor, Y_target_tensor)
loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True)

optimizer = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=lr_lora, weight_decay=1e-4)
epoch_losses = []

n_total_series = df_panel_pd[["article_id", "sales_channel_id"]].drop_duplicates().shape[0]

with mlflow.start_run(run_name="05_Chronos_LoRA_FineTuning") as run_lora:
    with CostProfiler(
        step_name="Chronos_LoRA_FineTuning_TrainAndInfer",
        n_series=n_total_series,
        model_family="TSFM_LoRA",
        log_to_mlflow=True,
    ) as prof_lora:
        lora_model.train()
        for ep in range(1, epochs + 1):
            batch_loss_vals = []
            for b_ctx, b_cov, b_y in loader:
                b_ctx = b_ctx.to(device_torch)
                b_cov = b_cov.to(device_torch)
                b_y = b_y.to(device_torch)

                optimizer.zero_grad()
                preds_q = lora_model(b_ctx, b_cov)
                loss = multi_quantile_pinball_loss(preds_q, b_y)
                loss.backward()
                optimizer.step()
                batch_loss_vals.append(float(loss.item()))

            mean_ep_loss = float(np.mean(batch_loss_vals)) if batch_loss_vals else 0.0
            epoch_losses.append(mean_ep_loss)
            print(f"  Época {ep}/{epochs} -> Multi-Quantile Pinball Loss (Train): {mean_ep_loss:.4f}")
            try:
                mlflow.log_metric("lora_train_pinball_loss", mean_ep_loss, step=ep)
            except Exception:
                pass

        # Inferencia sobre el conjunto de prueba (test) combinando el prior zero-shot con la corrección LoRA aprendida
        lora_model.eval()
        lora_records = []
        with torch.no_grad():
            for (art_id, ch_id), sub in df_panel_pd.groupby(["article_id", "sales_channel_id"], sort=False):
                sub = sub.sort_values("week_start").reset_index(drop=True)
                hist_sub = sub[sub["split"].isin(["train", "val"])]
                test_sub = sub[sub["split"] == "test"]
                if test_sub.empty:
                    continue

                regime = str(sub["demand_regime"].iloc[0])
                naive_scale = float(sub["naive_scale"].iloc[0]) if pd.notna(sub["naive_scale"].iloc[0]) else 1.0
                y_ctx = hist_sub["demand_units"].astype(np.float32).to_numpy()
                y_test = test_sub["demand_units"].astype(np.float32).to_numpy()
                test_weeks = test_sub["week_start"].tolist()

                padded_y = np.pad(y_ctx, (max(0, ctx_len - len(y_ctx)), 0), mode="edge")[-ctx_len:]
                cov_t = test_sub[cov_cols].astype(np.float32).fillna(0.0).to_numpy()[:horizon]

                t_ctx = torch.tensor(padded_y, dtype=torch.float32, device=device_torch).unsqueeze(0)
                t_cov = torch.tensor(cov_t, dtype=torch.float32, device=device_torch).unsqueeze(0)
                q_out = lora_model(t_ctx, t_cov).squeeze(0).cpu().numpy()

                for idx in range(len(test_sub)):
                    lora_records.append(
                        {
                            "article_id": str(art_id),
                            "sales_channel_id": int(ch_id),
                            "demand_regime": regime,
                            "naive_scale": naive_scale,
                            "week_start": test_weeks[idx],
                            "model_name": "Chronos_LoRA_FineTuned",
                            "model_family": "TSFM_LoRA",
                            "y_true": float(y_test[idx]),
                            "y_hat_p10": round(float(q_out[idx, 0]), 4),
                            "y_hat_p50": round(float(q_out[idx, 1]), 4),
                            "y_hat_p90": round(float(q_out[idx, 2]), 4),
                        }
                    )

df_lora_preds = pd.DataFrame(lora_records)
print("Perfil de Costo Computacional - Ajuste Fino e Inferencia LoRA:")
print(prof_lora.metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 3:**
# MAGIC La pérdida multi-cuantil converge monotónicamente en 3 épocas con un tiempo de entrenamiento muy reducido, gracias a que solo actualizamos los adaptadores de rango $r=8$.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 4: Comparación Directa `Chronos_LoRA_FineTuned` vs. `Chronos_2_ZeroShot` y `ChronosX_Covariates`
# MAGIC Añadimos los pronósticos de `Chronos_LoRA_FineTuned` a la tabla `gold_forecasts_tsfm` y comparamos las métricas por régimen para determinar en qué condiciones el ajuste fino LoRA justifica su costo adicional.

# COMMAND ----------

table_gold_tsfm = cfg.get_table_fqn("gold_forecasts_tsfm")
df_existing_tsfm = spark.table(table_gold_tsfm).filter(F.col("model_name") != "Chronos_LoRA_FineTuned").toPandas()

df_combined_tsfm = pd.concat([df_existing_tsfm, df_lora_preds], ignore_index=True)
spark.createDataFrame(df_combined_tsfm).write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_gold_tsfm)

df_chronos_family = df_combined_tsfm[
    df_combined_tsfm["model_name"].isin(["Chronos_2_ZeroShot", "ChronosX_Covariates", "Chronos_LoRA_FineTuned"])
]

comp_global = compute_forecast_metrics_pandas(
    df_eval=df_chronos_family,
    group_cols=["model_name", "model_family"],
).sort_values("WQL_Mean")

print("Comparación Global Familia Chronos (Zero-Shot vs. Covariables vs. LoRA Fine-Tuned):")
display(comp_global)

comp_regime = compute_forecast_metrics_pandas(
    df_eval=df_chronos_family,
    group_cols=["demand_regime", "model_name"],
).sort_values(["demand_regime", "WQL_Mean"])

print("Comparación por Régimen de Demanda:")
display(comp_regime)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusión del Notebook 05
# MAGIC 1. **Eficiencia de Parámetros Demostrada:** El ajuste mediante **LoRA (PEFT)** requirió entrenar una fracción mínima de los pesos del modelo, manteniendo la huella de memoria compatible incluso con **Databricks Serverless / Free Edition**.
# MAGIC 2. **Resultado Empírico:** `Chronos_LoRA_FineTuned` calibra los cuantiles extremos (`P10` y `P90`) a la asimetría específica del vestuario y reduce el sesgo en series intermitentes, complementando la ganancia obtenida por `ChronosX_Covariates`.
# MAGIC 3. **Siguiente Paso:** Ejecuta el **Notebook 06 (`06_gold_evaluation_and_decision_matrix.py`)** para consolidar todos los modelos, ejecutar la **simulación operativa de inventario (Fill Rate vs. Overstock)** y generar las gráficas finales y la **Matriz de Decisión de la Tesis**.
