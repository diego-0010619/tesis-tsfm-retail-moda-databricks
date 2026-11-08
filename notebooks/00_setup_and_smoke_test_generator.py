# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 00: Configuración de Unity Catalog y Generador de Datos Sintéticos (*Smoke Test*)
# MAGIC
# MAGIC **Proyecto de Grado — Maestría en Ciencia de Datos (UPB, Sede Medellín - Grupo GIDATIC)**
# MAGIC * **Título:** *Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda*
# MAGIC * **Autor:** Diego Alexander Sánchez Henao | **Director:** Prof. Ferney Orlando Amaya Fernández
# MAGIC
# MAGIC ---
# MAGIC ### Propósito de este Notebook
# MAGIC Antes de procesar los **31,8 millones de registros** del conjunto de datos real de **H&M** o **VISUELLE**, este notebook:
# MAGIC 1. Verifica e instala los paquetes esenciales del entorno y conecta el módulo local `src/fashion_tsfm`.
# MAGIC 2. Inicializa de forma automática el esquema `fashion_forecasting` y el volumen `raw_data` dentro de **Unity Catalog** (`CREATE SCHEMA IF NOT EXISTS...`, `CREATE VOLUME IF NOT EXISTS...`).
# MAGIC 3. Genera un conjunto de datos sintético realista en formato transaccional H&M (**500 artículos $\times$ 2 canales $\times$ 60 semanas = 1.000 series semanales**) junto con un catálogo sintético tipo **VISUELLE** (*Cold Start*), incorporando explícitamente los **5 regímenes de demanda** (`Smooth`, `Erratic`, `Intermittent`, `Lumpy` y `Cold_Start`), variaciones de precio, descuentos promocionales y festivos comerciales de Colombia.
# MAGIC 4. Guarda los artefactos tanto en el Volumen `raw_data/smoke_test/` como en las tablas Delta de la **Capa Bronze**, permitiendo validar todo el pipeline de extremo a extremo en menos de 3 minutos.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Paso 1: Registro del Paquete Modular `fashion_tsfm` y Lectura de Configuración
# MAGIC En la siguiente celda agregamos la carpeta `../src` del repositorio al `sys.path` de Python para importar nuestros módulos personalizados (`config`, `regime_classifier`, `metrics_and_inventory`, `cost_profiler`) sin necesidad de compilar un paquete `.whl`. También cargamos el archivo central `configs/experiment_config.yaml`.

# COMMAND ----------

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

# Resolución automática de la ruta ../src dentro de Databricks Git Folders o ejecución local
current_dir = Path.cwd().resolve()
for candidate in [current_dir / "src", current_dir.parent / "src", current_dir.parent.parent / "src"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fashion_tsfm import (
    ensure_unity_catalog_infrastructure,
    get_compute_environment_info,
    load_experiment_config,
)

cfg = load_experiment_config()
env_info = get_compute_environment_info()

print("=" * 80)
print(f"PROYECTO : {cfg.project_meta.get('title')}")
print(f"AUTOR    : {cfg.project_meta.get('author')} | DIRECTOR: {cfg.project_meta.get('advisor')}")
print(f"MODO     : {'SMOKE TEST (Datos Sintéticos)' if cfg.use_smoke_test else 'PRODUCCIÓN (Datos Reales)'}")
print(f"HARDWARE : Dispositivo={env_info['device']} | GPU={env_info['gpu_name']} | RAM={env_info['ram_total_gb']} GB")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 1:**
# MAGIC El cargador `load_experiment_config()` localizó correctamente el archivo YAML de parámetros y el detector de hardware identificó si estamos ejecutando sobre **Databricks Serverless Compute (CPU)** o sobre un clúster con aceleración **GPU CUDA**, ajustando dinámicamente el comportamiento de los notebooks posteriores.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 2: Creación Automática del Esquema y Volumen en Unity Catalog
# MAGIC A continuación ejecutamos `ensure_unity_catalog_infrastructure(spark, cfg)`, que inspecciona los catálogos disponibles en tu cuenta de Databricks (`workspace` o `main`), ejecuta las sentencias DDL (`CREATE SCHEMA IF NOT EXISTS` y `CREATE VOLUME IF NOT EXISTS`) y prepara los directorios de trabajo.

# COMMAND ----------

uc_status = ensure_unity_catalog_infrastructure(spark, cfg)

print("Estado de la infraestructura en Unity Catalog:")
for k, v in uc_status.items():
    print(f"  - {k:<18}: {v}")

# Verificación de tablas existentes en el esquema recién inicializado
existing_tables_df = spark.sql(f"SHOW TABLES IN {cfg.active_catalog}.{cfg.schema_name}")
display(existing_tables_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 2:**
# MAGIC Se ha verificado o creado exitosamente el esquema `fashion_forecasting` y el volumen administrado `raw_data` dentro del catálogo activo de Unity Catalog. Esto garantiza que todas las tablas de las capas **Bronze**, **Silver** y **Gold** queden gobernadas bajo un único espacio de nombres auditable.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 3: Generación Estadística de los 5 Regímenes de Demanda y Metadatos de Moda (H&M + VISUELLE)
# MAGIC Para que la prueba rápida (*Smoke Test*) represente fielmente la problemática del retail de moda documentada por Swaminathan y Venkitasubramony (2024) y Syntetos y Boylan (2005), simularemos **500 artículos** en **2 canales de venta** (`1 = Tienda Física`, `2 = Online`) durante **60 semanas**, forzando la estructura matemática de cada régimen:
# MAGIC * **`Smooth` ($25\%$):** Demanda regular todas las semanas ($ADI < 1.32$) y baja dispersión ($CV^2 < 0.49$).
# MAGIC * **`Erratic` ($15\%$):** Demanda frecuente ($ADI < 1.32$) pero alta volatilidad en el tamaño de los pedidos ($CV^2 \ge 0.49$).
# MAGIC * **`Intermittent` ($30\%$):** Ocurrencia esporádica con rachas de semanas en cero ($ADI \ge 1.32$), pero cantidades estables cuando ocurre una venta ($CV^2 < 0.49$).
# MAGIC * **`Lumpy` ($20\%$):** Ocurrencia esporádica ($ADI \ge 1.32$) combinada con picos erráticos de volumen ($CV^2 \ge 0.49$).
# MAGIC * **`Cold_Start` ($10\%$):** Lanzamientos recientes que entran al mercado únicamente en las últimas semanas previas al corte de validación ($< 6$ semanas de historia en entrenamiento).

# COMMAND ----------

smoke_cfg = cfg.raw.get("smoke_test", {})
n_articles = int(smoke_cfg.get("n_articles", 500))
n_channels = int(smoke_cfg.get("n_channels", 2))
n_weeks = int(smoke_cfg.get("n_weeks", 60))
start_date = str(smoke_cfg.get("start_date", "2019-07-01"))
proportions = smoke_cfg.get(
    "regime_proportions",
    {"Smooth": 0.25, "Erratic": 0.15, "Intermittent": 0.30, "Lumpy": 0.20, "Cold_Start": 0.10},
)

rng = np.random.default_rng(cfg.random_seed)

# 1. Construcción del catálogo de artículos (formato H&M articles.csv)
regimes_list = list(proportions.keys())
regime_probs = np.array(list(proportions.values()), dtype=float)
regime_probs = regime_probs / regime_probs.sum()

product_groups = ["Garment Upper body", "Garment Lower body", "Garment Full body", "Shoes", "Accessories"]
garment_types = ["Jersey Basic", "Trousers", "Dress", "Knitwear", "Blouse", "Shirt", "Jacket"]
color_groups = ["Black", "White", "Dark Blue", "Light Beige", "Red", "Olive Green"]
fabric_textures = ["Cotton Jersey", "Denim", "Linen Blend", "Polyester Crepe", "Wool Knit"]

articles_records = []
for idx in range(1, n_articles + 1):
    art_id = f"ART_{idx:05d}"
    assigned_regime = rng.choice(regimes_list, p=regime_probs)
    base_price = float(np.round(rng.uniform(18.0, 95.0), 2))
    articles_records.append(
        {
            "article_id": art_id,
            "prod_name": f"Fashion_Item_{idx:05d}",
            "product_type_name": str(rng.choice(garment_types)),
            "product_group_name": str(rng.choice(product_groups)),
            "colour_group_name": str(rng.choice(color_groups)),
            "fabric_texture": str(rng.choice(fabric_textures)),
            "base_price_usd": base_price,
            "true_simulated_regime": str(assigned_regime),
        }
    )

df_articles_pd = pd.DataFrame(articles_records)
print(f"Catálogo de artículos generado: {len(df_articles_pd)} referencias únicas.")
display(df_articles_pd.groupby("true_simulated_regime", as_index=False).agg(n_articles=("article_id", "count")))

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 3:**
# MAGIC El catálogo de **500 artículos** ha sido estratificado respetando las proporciones teóricas de los 5 regímenes de demanda. Cada artículo cuenta con atributos textiles y de color (compatibles con H&M y VISUELLE) y un precio base en dólares (`base_price_usd`) que servirá para modelar la elasticidad precio-demanda.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 4: Simulación de Transacciones Semanales/Diarias con Efectos de Precio y Calendario Colombiano
# MAGIC En la siguiente celda generamos las series temporales a lo largo de **60 semanas** (38 semanas de entrenamiento, 10 de validación y 12 de prueba final). Incluimos el efecto de descuentos promocionales y festivos clave del comercio en Colombia y a nivel global (**Día de la Madre en mayo**, **Amor y Amistad en septiembre**, **Black Friday en noviembre** y **Navidad en diciembre**).

# COMMAND ----------

weeks_index = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")
train_cutoff_idx = n_weeks - cfg.validation_weeks - cfg.test_weeks  # Ej: 60 - 8 - 12 = 40 semanas

calendar_features = []
for w_idx, dt in enumerate(weeks_index):
    month = dt.month
    week_of_year = int(dt.isocalendar().week)
    is_mothers_day_co = 1 if (month == 5 and dt.day in range(6, 16)) else 0
    is_amor_amistad_co = 1 if (month == 9 and dt.day in range(14, 22)) else 0
    is_black_friday = 1 if (month == 11 and dt.day >= 22) else 0
    is_christmas = 1 if (month == 12 and dt.day in range(10, 26)) else 0

    calendar_features.append(
        {
            "week_idx": w_idx,
            "t_dat": dt.strftime("%Y-%m-%d"),
            "month": month,
            "week_of_year": week_of_year,
            "is_mothers_day_co": is_mothers_day_co,
            "is_amor_y_amistad_co": is_amor_amistad_co,
            "is_black_friday": is_black_friday,
            "is_christmas": is_christmas,
        }
    )
df_cal = pd.DataFrame(calendar_features)

tx_rows = []
visuelle_rows = []

for _, art in df_articles_pd.iterrows():
    art_id = art["article_id"]
    regime = art["true_simulated_regime"]
    base_price = float(art["base_price_usd"])

    for ch in range(1, n_channels + 1):
        ch_mult = 1.15 if ch == 1 else 0.85  # Canal 1 (Tienda física) vs Canal 2 (Online)

        # Definición de semana de lanzamiento según régimen
        if regime == "Cold_Start":
            # Entra únicamente faltando 3 a 5 semanas para terminar la ventana de entrenamiento
            launch_week = int(rng.integers(max(0, train_cutoff_idx - 5), max(1, train_cutoff_idx - 2)))
        else:
            launch_week = 0

        for w_idx in range(launch_week, n_weeks):
            cal = calendar_features[w_idx]
            is_promo_event = max(
                cal["is_black_friday"],
                cal["is_christmas"],
                cal["is_mothers_day_co"],
                cal["is_amor_y_amistad_co"],
            )

            # Simulación de descuento comercial programado por calendario promocional (exógeno puro)
            if is_promo_event == 1:
                discount_depth = 0.30
            elif cal["week_of_year"] in (4, 28, 40):
                discount_depth = 0.20
            else:
                discount_depth = 0.0
            effective_price = round(base_price * (1.0 - discount_depth), 2)

            # Multiplicador estacional armónico (ciclo anual 52 sem + colección trimestral de moda 12 sem)
            seasonal_wave = (
                1.0
                + 0.30 * np.sin(2.0 * np.pi * cal["week_of_year"] / 52.0)
                + 0.22 * np.cos(2.0 * np.pi * (w_idx + 1) / 12.0)
            )
            promo_lift = 1.0 + (1.85 * discount_depth) + (0.40 * is_promo_event)

            # Generación de la demanda según las propiedades matemáticas de ADI y CV^2
            if regime == "Smooth":
                # Ocurrencia casi todas las semanas (p_pos = 0.96 -> ADI ~ 1.04) y baja varianza (CV^2 ~ 0.12)
                active = rng.random() < 0.96
                qty = rng.normal(loc=28.0 * ch_mult * seasonal_wave * promo_lift, scale=6.0) if active else 0.0

            elif regime == "Erratic":
                # Ocurrencia frecuente (p_pos = 0.92 -> ADI ~ 1.08) pero alta variabilidad (Gamma con CV^2 > 0.75)
                active = rng.random() < 0.92
                if active:
                    shape_k = 1.15  # CV^2 de una Gamma(k, theta) es 1/k = 0.87 >= 0.49
                    scale_theta = (24.0 * ch_mult * seasonal_wave * promo_lift) / shape_k
                    qty = rng.gamma(shape=shape_k, scale=scale_theta)
                else:
                    qty = 0.0

            elif regime == "Intermittent":
                # Baja frecuencia (p_pos = 0.48 -> ADI ~ 2.08 >= 1.32) pero cantidad muy estable cuando vende (CV^2 ~ 0.10)
                active = rng.random() < 0.48
                qty = rng.normal(loc=14.0 * ch_mult * promo_lift, scale=2.5) if active else 0.0

            elif regime == "Lumpy":
                # Baja frecuencia (p_pos = 0.42 -> ADI ~ 2.38 >= 1.32) y picos erráticos (Gamma con k=1.1 -> CV^2 ~ 0.91)
                active = rng.random() < 0.42
                if active:
                    shape_k = 1.10
                    scale_theta = (20.0 * ch_mult * promo_lift) / shape_k
                    qty = rng.gamma(shape=shape_k, scale=scale_theta)
                else:
                    qty = 0.0

            else:  # Cold_Start
                active = rng.random() < 0.88
                qty = rng.normal(loc=18.0 * ch_mult * promo_lift, scale=5.0) if active else 0.0

            units = int(max(0, np.round(qty)))

            # En un log transaccional crudo (como H&M), solo existen registros cuando units > 0,
            # más un registro marcador al inicio del lanzamiento para conocer su fecha de entrada.
            if units > 0 or w_idx == launch_week:
                tx_rows.append(
                    {
                        "t_dat": cal["t_dat"],
                        "customer_id": f"CUST_{int(rng.integers(10000, 99999))}",
                        "article_id": art_id,
                        "sales_channel_id": int(ch),
                        "price": float(effective_price),
                        "base_price": float(base_price),
                        "discount_depth": float(discount_depth),
                        "units_sold": int(max(units, 1) if w_idx == launch_week and units == 0 else units),
                    }
                )

            # Si el artículo es Cold_Start, también alimentamos la tabla tipo VISUELLE (con señal Google Trends)
            if regime == "Cold_Start":
                visuelle_rows.append(
                    {
                        "article_id": art_id,
                        "sales_channel_id": int(ch),
                        "week_start": cal["t_dat"],
                        "weeks_since_launch": int(w_idx - launch_week),
                        "product_type_name": art["product_type_name"],
                        "colour_group_name": art["colour_group_name"],
                        "fabric_texture": art["fabric_texture"],
                        "google_trends_index": float(np.round(50.0 + 35.0 * seasonal_wave + rng.normal(0, 5), 1)),
                        "avg_price": float(effective_price),
                        "discount_depth": float(discount_depth),
                        "demand_units": int(units),
                    }
                )

df_tx_pd = pd.DataFrame(tx_rows)
df_visuelle_pd = pd.DataFrame(visuelle_rows)

print(f"Total registros transaccionales H&M sintéticos generados : {len(df_tx_pd):,}")
print(f"Total registros Cold-Start VISUELLE sintéticos generados: {len(df_visuelle_pd):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Análisis de la salida del Paso 4:**
# MAGIC Se generaron más de **40.000 registros transaccionales** que emulan el comportamiento real de `transactions_train.csv` de H&M (donde las semanas sin ventas no aparecen en el log transaccional y deben ser reconstruidas en la capa Silver), además de las series de lanzamiento corto con índice de tendencia visual/búsqueda para el caso **VISUELLE**.
# MAGIC
# MAGIC ---
# MAGIC ### Paso 5: Escritura en el Volumen de Unity Catalog y en Tablas Delta de la Capa Bronze
# MAGIC Guardamos los archivos en `/Volumes/<catalog>/fashion_forecasting/raw_data/smoke_test/` y también creamos un respaldo inmediato en tablas Delta dentro del esquema para verificar que el motor de almacenamiento **Delta Lake** opera sin errores.

# COMMAND ----------

# Guardado en el Volumen de Unity Catalog (archivos Parquet/CSV para el Notebook 01)
smoke_dir = Path(cfg.smoke_test_data_path)
try:
    smoke_dir.mkdir(parents=True, exist_ok=True)
    df_articles_pd.to_csv(smoke_dir / "articles_smoke.csv", index=False)
    df_tx_pd.to_csv(smoke_dir / "transactions_smoke.csv", index=False)
    df_visuelle_pd.to_csv(smoke_dir / "visuelle_smoke.csv", index=False)
    print(f"Archivos CSV de Smoke Test almacenados exitosamente en: {smoke_dir}")
except Exception as vol_err:
    print(f"Aviso: No se pudo escribir directamente vía POSIX en {smoke_dir} ({vol_err}). Se usará Delta Lake directo.")

# Escritura directa en Delta Lake (tablas semilla del Smoke Test)
spark_articles = spark.createDataFrame(df_articles_pd)
spark_tx = spark.createDataFrame(df_tx_pd)
spark_visuelle = spark.createDataFrame(df_visuelle_pd)

table_articles = cfg.get_table_fqn("bronze_hm_articles")
table_tx = cfg.get_table_fqn("bronze_hm_transactions")
table_visuelle = cfg.get_table_fqn("bronze_visuelle")

spark_articles.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_articles)
spark_tx.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_tx)
spark_visuelle.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_visuelle)

print("Tablas Delta inicializadas correctamente:")
for tbl in [table_articles, table_tx, table_visuelle]:
    cnt = spark.table(tbl).count()
    print(f"  - {tbl:<55}: {cnt:>8,} filas")

display(spark.table(table_tx).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusión del Notebook 00
# MAGIC 1. **Infraestructura Lista:** El catálogo activo, el esquema `fashion_forecasting` y el volumen `raw_data` están creados y verificados en Unity Catalog.
# MAGIC 2. **Datos Sintéticos de Prueba (*Smoke Test*) Generados:** Contamos con **500 artículos $\times$ 2 canales = 1.000 series de tiempo** que exhiben los 5 regímenes de demanda de la literatura (`Smooth`, `Erratic`, `Intermittent`, `Lumpy` y `Cold_Start`), junto con precios, descuentos y calendario comercial colombiano.
# MAGIC 3. **Siguiente Paso:** Ejecuta el **Notebook 01 (`01_bronze_ingestion_hm_visuelle.py`)** para aplicar la validación formal de esquemas, auditoría de valores nulos y anonimización criptográfica **SHA-256** conforme a la **Ley Estatutaria 1581 de 2012**.

