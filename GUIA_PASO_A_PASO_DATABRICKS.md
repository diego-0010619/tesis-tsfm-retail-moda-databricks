# Guía Paso a Paso Desde Cero: GitHub + Databricks Lakehouse para tu Proyecto de Grado (UPB)

**Autor de la Tesis:** Diego Alexander Sánchez Henao  
**Director Metodológico:** Prof. Ferney Orlando Amaya Fernández  
**Programa:** Maestría en Ciencia de Datos — Universidad Pontificia Bolivariana (UPB), Sede Medellín (Grupo GIDATIC)

Esta guía ha sido diseñada especialmente para quien **utiliza Databricks por primera vez**. No asume ningún conocimiento previo de la interfaz de Databricks ni de integración con GitHub: te indica exactamente **en qué menú, botón, ícono o pestaña hacer clic** en cada momento.

---

## Fase 1: Preparar y Subir los Artefactos a tu Repositorio de GitHub

Todos los archivos del proyecto ya están creados en la carpeta de tu computador (`e:\Users\user\Desktop\Maestría ciencia de datos\Proyecto de grado`). Ahora los subiremos a tu cuenta de **GitHub** para que **Databricks** pueda sincronizarlos automáticamente.

### Paso 1.1: Crear el repositorio vacío en GitHub.com
1. Abre tu navegador web e ingresa a [https://github.com](https://github.com) e inicia sesión con tu cuenta.
2. En la esquina superior derecha de la pantalla, haz clic en el ícono **`+`** (al lado de tu foto de perfil) y selecciona **`New repository`**.
3. En el formulario de creación:
   * **Repository name:** Escribe exactamente `tesis-tsfm-retail-moda-databricks`
   * **Description (opcional):** `Modelos fundacionales de series de tiempo a escala Big Data para el pronóstico de demanda en el retail de moda - Maestría en Ciencia de Datos UPB`
   * **Visibilidad:** Selecciona **`Private`** (si deseas mantener tu código privado mientras redactas la tesis) o **`Public`**.
   * **IMPORTANTE:** **NO** marques las casillas *"Add a README file"*, *"Add .gitignore"* ni *"Choose a license"* (porque Antigravity ya creó esos archivos localmente para ti).
4. Haz clic en el botón verde **`Create repository`** abajo a la derecha.
5. Copia la URL HTTPS que aparece en pantalla (tendrá la forma `https://github.com/TU_USUARIO/tesis-tsfm-retail-moda-databricks.git`).

### Paso 1.2: Subir los archivos desde tu computador hacia GitHub
Abre la terminal integrada de **Antigravity** (o PowerShell) ubicada en la carpeta del proyecto (`e:\Users\user\Desktop\Maestría ciencia de datos\Proyecto de grado`) y ejecuta uno por uno los siguientes comandos (reemplazando `TU_USUARIO` por tu nombre de usuario real de GitHub):

```powershell
# 1. Inicializar el repositorio Git local
git init

# 2. Renombrar la rama principal a 'main'
git branch -M main

# 3. Agregar todos los artefactos generados (respetando el .gitignore)
git add .

# 4. Crear el primer commit con toda la estructura de la tesis
git commit -m "feat: arquitectura Medallion, paquete fashion_tsfm y 7 notebooks Databricks para tesis UPB"

# 5. Vincular tu carpeta local con el repositorio remoto en GitHub
git remote add origin https://github.com/diego-0010619/tesis-tsfm-retail-moda-databricks.git

# 6. Subir (push) los archivos a GitHub
git push -u origin main
```

### Paso 1.3: Generar tu Personal Access Token (PAT) en GitHub para Databricks
Para que Databricks pueda leer y guardar cambios en tu repositorio de GitHub de forma segura, necesita una "llave" llamada **Personal Access Token (PAT)**:

1. En **GitHub.com**, haz clic en tu **foto de perfil** (esquina superior derecha) y selecciona **`Settings`** (Configuración).
2. Desplázate hasta el final de la barra lateral izquierda y haz clic en **`<> Developer settings`** (es la última opción abajo a la izquierda).
3. En el menú izquierdo, haz clic en **`Personal access tokens`** y elige **`Tokens (classic)`** (es el método más directo y universalmente compatible con Databricks) o **`Fine-grained tokens`**:
   * **Si eliges `Tokens (classic)` (Recomendado por simplicidad):**
     1. Haz clic en el botón **`Generate new token`** $\rightarrow$ **`Generate new token (classic)`**.
     2. En **Note**, escribe: `Token Databricks Tesis UPB`.
     3. En **Expiration**, selecciona `90 days` o `Custom` (hasta la fecha de sustentación de tu tesis).
     4. En la lista de casillas (**Select scopes**), marca la casilla principal **`repo`** (se marcarán automáticamente sus sub-casillas `repo:status`, `repo_deployment`, `public_repo`, etc.) y marca también **`workflow`**.
     5. Baja hasta el final y haz clic en el botón verde **`Generate token`**.
   * **Si eliges `Fine-grained tokens`:**
     1. Haz clic en **`Generate new token`**, ponle nombre `Token Databricks Tesis UPB` y en **Repository access** elige *All repositories* (o *Only select repositories* seleccionando `tesis-tsfm-retail-moda-databricks`).
     2. Despliega **`Repository permissions`** y en **`Contents`** cambia el permiso de *No access* a **`Read and write`**. En **`Metadata`** déjalo en *Read-only*.
     3. Haz clic en **`Generate token`**.
4. **¡MUY IMPORTANTE!** Copia inmediatamente el código del token que empieza por `ghp_...` o `github_pat_...` y guárdalo temporalmente en un bloc de notas seguro, porque **GitHub no lo volverá a mostrar** cuando cierres esa página.

---

## Fase 2: Primer Ingreso y Recorrido Guiado por la Interfaz de Databricks

### Paso 2.1: Qué ves al iniciar sesión en Databricks
Cuando ingresas a tu cuenta de **Databricks** (por ejemplo, en [Databricks Free Edition / Serverless](https://login.databricks.com/) o en el workspace académico/empresarial que utilices), verás una **barra lateral izquierda oscura o gris clara** con varios íconos principales. Esto es lo que hace cada uno:

| Menú en la Barra Izquierda | ¿Para qué sirve en tu Tesis? |
| :--- | :--- |
| **`+ New`** (Botón superior) | Permite crear rápidamente un nuevo Notebook, un Job o una carpeta Git. |
| **`Workspace`** (Ícono de carpetas) | Es tu **Explorador de Archivos de Código**. Aquí vivirán tus notebooks (`00` al `06`) y el código Python sincronizado con tu repositorio de GitHub dentro de `Workspace -> Users -> tu_correo`. |
| **`Catalog`** (Ícono de cilindros/base de datos) | Es el **Explorador de Datos (Unity Catalog)**. Aquí verás visualmente tu esquema `fashion_forecasting`, el volumen `raw_data` (donde se suben los archivos CSV) y las tablas creadas de las capas `bronze`, `silver` y `gold`. |
| **`Workflows` / `Jobs`** (Ícono de diagrama) | Permite programar la ejecución automática en cadena de tus notebooks (`00 -> 01 -> 02 -> ... -> 06`). |
| **`Compute`** (Ícono de nube/servidor) | Muestra las máquinas virtuales o motores de cómputo disponibles para ejecutar tu código. |
| **`Experiments`** (Ícono de matraz de laboratorio / MLflow) | Es el panel de **MLflow** (bajo la sección *Machine Learning*). Aquí se registran automáticamente las tablas comparativas de WAPE, MASE, WQL, tiempos de ejecución y costo computacional por cada 1.000 series. |

### Paso 2.2: Diferencia entre *Serverless Compute* y un Clúster Propio (y el botón *Connect*)
* **Serverless Compute (Predeterminado en Databricks Free Edition):**
  * No tienes que configurar ni encender ningún servidor manualmente.
  * Cuando abres un notebook, en la **esquina superior derecha** verás un botón que dice **`Connect`** (o un círculo verde con la palabra **`Serverless`**).
  * Al hacer clic en **`Serverless`** (o al ejecutar la primera celda), Databricks te asigna instantáneamente en 3 a 5 segundos un entorno con Python y Apache Spark listo para trabajar.
* **Clúster Propio (En cuentas estándar/empresariales de AWS, Azure o GCP con GPU):**
  * Si más adelante deseas correr el ajuste completo con **GPU NVIDIA (T4 o A10G)** en una cuenta paga o con créditos académicos, vas a **`Compute`** $\rightarrow$ **`Create compute`**, eliges un **Databricks Runtime ML** (ej. `15.4 LTS ML (GPU, Scala 2.12, Spark 3.5.0)`) y luego en tu notebook haces clic en **`Connect`** arriba a la derecha para elegir ese clúster.
  * **Ventaja de tu código:** Los notebooks que generamos detectan automáticamente (`torch.cuda.is_available()`) si estás en **Serverless CPU** o en **GPU**, por lo que **funcionan sin cambiar una sola línea de código en ambos entornos**.

---

## Fase 3: Conectar GitHub con Databricks y Traer los Artefactos (Git Folders / Repos)

### Paso 3.1: Vincular tu Personal Access Token (PAT) de GitHub en Databricks
1. Dentro de Databricks, mira hacia la **esquina superior derecha** y haz clic sobre tu **ícono de perfil** (el círculo con la inicial de tu nombre o tu correo electrónico).
2. En el menú desplegable, haz clic en **`Settings`** (Configuración).
3. En la columna izquierda de la pantalla de configuración (bajo la sección *User*), haz clic en **`Linked accounts`** (Cuentas vinculadas).
4. Verás la sección **Git integration**:
   * **Git provider:** Selecciona **`GitHub`**.
   * Marca la opción **`Personal access token`**.
   * **GitHub username or email:** Escribe tu nombre de usuario de GitHub.
   * **Token:** Pega el **Personal Access Token (`ghp_...`)** que copiaste en el Paso 1.3.
5. Haz clic en el botón azul **`Save`**. Aparecerá un mensaje verde confirmando que tus credenciales de Git quedaron guardadas.

### Paso 3.2: Clonar el repositorio de GitHub dentro de tu Workspace (Git Folder)
1. En la barra lateral izquierda de Databricks, haz clic en **`Workspace`**.
2. Haz clic en la carpeta **`Users`** y luego haz clic sobre la carpeta que tiene **tu correo electrónico** (ej. `diego.sanchez@...`).
3. En la esquina superior derecha de esa vista, haz clic en el botón azul **`Create`** y selecciona **`Git folder`** (en versiones anteriores llamado *Repo*).
4. Se abrirá una ventana emergente con los siguientes campos:
   * **Git repository URL:** Pega la URL de tu repositorio de GitHub:  
     `https://github.com/diego-0010619/tesis-tsfm-retail-moda-databricks.git`
   * **Git provider:** Se seleccionará automáticamente `GitHub`.
   * **Git folder name:** Se llenará automáticamente como `tesis-tsfm-retail-moda-databricks`.
5. Haz clic en el botón azul **`Create Git folder`**.
6. ¡Listo! En pocos segundos verás aparecer dentro de Databricks todas tus carpetas (`configs/`, `notebooks/`, `src/`, `README.md`, `requirements.txt` y esta guía).

### Paso 3.3: Cómo hacer `Pull` (traer cambios de GitHub) y `Commit & Push` (guardar desde Databricks a GitHub)
* **Para traer a Databricks cambios que hayas hecho en tu PC con Antigravity (`Pull`):**
  1. Entra a tu carpeta `tesis-tsfm-retail-moda-databricks` en Databricks.
  2. Haz clic en el botón de la rama **`main`** (con el ícono de ramita de Git, situado junto al nombre del repositorio en la parte superior).
  3. Se abrirá el panel de Git: haz clic en el botón **`Pull`** (esquina superior derecha del cuadro emergente) para actualizar todos los archivos al instante.
* **Para subir a GitHub cambios que hayas editado directamente en Databricks (`Commit & Push`):**
  1. Haz clic en el mismo botón de la rama **`main`**.
  2. Verás a la izquierda la lista de archivos modificados. Escribe un mensaje descriptivo en la caja **Summary (required)** (ej. `" Ajuste de hiperparámetros en experimento"`).
  3. Haz clic en el botón azul **`Commit & Push`** abajo a la derecha.

---

## Fase 4: Configuración de Datos en Unity Catalog Volumes (Prueba Sintética vs. Datos Reales)

### Paso 4.1: Ejecutar tu primer Notebook (`00_setup_and_smoke_test_generator.py`)
1. En Databricks, ve a **`Workspace`** $\rightarrow$ **`Users`** $\rightarrow$ **`tu_correo`** $\rightarrow$ **`tesis-tsfm-retail-moda-databricks`** $\rightarrow$ **`notebooks`**.
2. Haz clic sobre **`00_setup_and_smoke_test_generator`**.
3. Mira arriba a la derecha del notebook:
   * Si dice **`Connect`**, haz clic ahí y selecciona **`Serverless`** (o tu clúster activo). Espera unos segundos hasta que aparezca un círculo verde.
4. **Instalar las librerías del proyecto en la sesión (opcional si falta alguna dependencia):**
   * Puedes insertar una celda al inicio o ejecutar en cualquier celda:
     ```python
     %pip install -r ../requirements.txt
     dbutils.library.restartPython()
     ```
   * *(Nota: Los notebooks `00` a `06` están programados de forma que las librerías base de Databricks Runtime ML funcionan de inmediato, pero correr `%pip install -r ../requirements.txt` instala paquetes específicos como `chronos-forecasting`, `statsforecast` y `peft`).*
5. Para ejecutar el notebook:
   * **Opción A (Celda por celda, recomendada para aprender):** Haz clic dentro de la primera celda de código y presiona **`Shift + Enter`** en tu teclado. Cuando termine (verás un chulito verde y el tiempo en segundos), lee el análisis inferior y presiona **`Shift + Enter`** en la siguiente celda.
   * **Opción B (Todo de una vez):** Haz clic en el botón **`Run all`** situado en la barra superior del notebook.

### Paso 4.2: Verificar visualmente tus Tablas y Volumen en la pestaña `Catalog`
Una vez que el Notebook `00` termine de ejecutarse (tarda menos de 2 minutos):
1. Haz clic en **`Catalog`** en la barra lateral izquierda de Databricks.
2. En el árbol izquierdo del explorador de catálogo, despliega tu catálogo activo (**`workspace`** o **`main`**).
3. Despliega el esquema **`fashion_forecasting`**.
4. Verás dos secciones:
   * **`Tables`:** Al desplegarla, verás tus primeras tablas Delta creadas: `bronze_hm_articles`, `bronze_hm_transactions` y `bronze_visuelle_coldstart`. Si haces clic en cualquiera de ellas, podrás explorar sus columnas, tipos de datos y una vista previa (*Sample Data*).
   * **`Volumes`:** Al desplegarla, verás el volumen **`raw_data`** y dentro la subcarpeta **`smoke_test`** con los archivos CSV generados.

### Paso 4.3: Cómo cargar después los Archivos Reales de Kaggle (H&M / VISUELLE / M5)
Cuando ya hayas probado todo el flujo con el *Smoke Test* y quieras correr con los datos reales de H&M o VISUELLE, tienes **dos formas** de cargarlos al Volumen `/Volumes/workspace/fashion_forecasting/raw_data/`:

* **Método A — Subida gráfica directa (Ideal para archivos medianos como `articles.csv`, `customers.csv` o `VISUELLE`):**
  1. Ve a **`Catalog`** en el menú izquierdo $\rightarrow$ **`workspace`** $\rightarrow$ **`fashion_forecasting`** $\rightarrow$ **`Volumes`** $\rightarrow$ **`raw_data`**.
  2. Haz clic en el botón azul **`Upload to this volume`** (esquina superior derecha).
  3. Arrastra los archivos desde tu computador y haz clic en **`Upload`**.

* **Método B — Descarga directa desde Kaggle hacia Databricks sin usar tu internet local (Ideal para `transactions_train.csv` de 31,8 millones de filas / 3.5 GB):**
  1. En tu cuenta de [Kaggle.com](https://www.kaggle.com/settings) $\rightarrow$ *Settings* $\rightarrow$ *API* $\rightarrow$ haz clic en *Create New Token* para obtener tu `kaggle.json` (`username` y `key`).
  2. Crea una celda temporal en Databricks y ejecuta:
     ```python
     %pip install kaggle
     import os
     os.environ["KAGGLE_USERNAME"] = "TU_USUARIO_KAGGLE"
     os.environ["KAGGLE_KEY"] = "TU_KEY_KAGGLE"

     # Descarga directa desde los servidores de Kaggle al Volumen de Databricks a velocidad de fibra en la nube:
     !mkdir -p /Volumes/workspace/fashion_forecasting/raw_data/hm
     !kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f transactions_train.csv -p /Volumes/workspace/fashion_forecasting/raw_data/hm
     !kaggle competitions download -c h-and-m-personalized-fashion-recommendations -f articles.csv -p /Volumes/workspace/fashion_forecasting/raw_data/hm
     !unzip -o /Volumes/workspace/fashion_forecasting/raw_data/hm/*.zip -d /Volumes/workspace/fashion_forecasting/raw_data/hm/
     ```

---

## Fase 5: Ejecución Paso a Paso de las Primeras Pruebas y Visualización en MLflow

### Paso 5.1: Orden Exacto de Ejecución en Modo *Smoke Test* (Datos Sintéticos)
Con `use_smoke_test: true` (que ya viene activado por defecto en `configs/experiment_config.yaml`), abre y ejecuta con **`Run all`** (o `Shift + Enter`) los notebooks en este orden numérico estricto:

1. **`00_setup_and_smoke_test_generator.py`**: Crea el esquema, volumen y las 1.000 series sintéticas con los 5 regímenes de demanda.
2. **`01_bronze_ingestion_hm_visuelle.py`**: Valida tipos de datos, limpia nulos y anonimiza `customer_id` con SHA-256 (Ley 1581 de 2012).
3. **`02_silver_weekly_panel_and_regimes.py`**: Construye el panel semanal balanceado, rellena semanas con demanda `0`, crea las covariables de precio/descuento/festivos de Colombia y clasifica cada serie en la matriz Syntetos-Boylan ($ADI$, $CV^2$).
4. **`03_baselines_stats_and_lightgbm.py`**: Ejecuta en paralelo `CrostonClassic`, `CrostonSBA`, `TSB`, `AutoETS`, `SeasonalNaive` y `LightGBM_Global` (cuantiles P10, P50, P90), guardando en `gold_forecasts_baselines`.
5. **`04_tsfm_zeroshot_and_covariates.py`**: Ejecuta los Modelos Fundacionales (`Chronos-Bolt`, `Chronos-2`, `TimesFM`, `Moirai`) en modo *Zero-Shot* y con covariables exógenas (`ChronosX_Covariates`, `Moirai_Covariates`), además de las curvas de escalamiento en $100$, $1.000$, $10.000$ y $100.000$ series.
6. **`05_tsfm_lora_finetuning.py`**: Ajusta eficientemente los adaptadores **LoRA ($r=8$)** sobre las series de entrenamiento y genera los pronósticos de `Chronos_LoRA_FineTuned`.
7. **`06_gold_evaluation_and_decision_matrix.py`**: Consolida los 13 modelos, ejecuta la simulación de inventario (*Fill Rate* vs. *Overstock*), genera las 4 figuras científicas de tesis y construye la tabla prescriptiva `gold_decision_matrix`.

### Paso 5.2: Cómo pasar del *Smoke Test* al Experimento Real Completo
Cuando hayas subido los archivos reales al volumen `raw_data`:
1. En Databricks, abre el archivo `configs/experiment_config.yaml`.
2. Busca la línea 53 en la sección `execution:` y cambia:
   ```yaml
   execution:
     use_smoke_test: false
   ```
3. Guarda el archivo (`Ctrl + S`) y ejecuta los notebooks **`01` al `06`**.

### Paso 5.3: Cómo Inspeccionar MLflow (`Experiments`) y Descargar Gráficas/Tablas para tu Documento de Tesis UPB
1. **Ver las corridas en MLflow:**
   * En la barra lateral izquierda de Databricks, haz clic en **`Experiments`** (ícono de matraz químico).
   * Haz clic en el experimento **`tesis_tsfm_retail_moda_upb`** (o el experimento asociado a cada notebook).
   * Verás cada corrida (*Run*) con columnas para `wall_clock_sec`, `sec_per_1000_series`, `cost_usd_per_1000`, `ram_mb` y `vram_peak_mb`. Puedes seleccionar varias corridas y hacer clic en **`Compare`** para generar gráficos automáticos de barras y dispersión.
2. **Descargar las Tablas en CSV / Excel para tus capítulos de la tesis:**
   * En cualquier tabla mostrada con `display(...)` en los notebooks `02`, `03`, `04`, `05` o `06`, pasa el cursor por la esquina inferior izquierda de la tabla, haz clic en la flecha junto a **`Table`** y selecciona **`Download as CSV`** o **`Download as Excel`**.
3. **Descargar las Figuras en Alta Resolución (300 DPI):**
   * En las figuras generadas al final de los notebooks `02` y `06`, puedes pasar el mouse sobre la esquina superior derecha de la imagen y hacer clic en el ícono de cámara/descarga para bajar el PNG, o ir directamente a **`Catalog`** $\rightarrow$ **`workspace`** $\rightarrow$ **`fashion_forecasting`** $\rightarrow$ **`Volumes`** $\rightarrow$ **`raw_data`** $\rightarrow$ **`thesis_figures`** y descargar los archivos `fig01_syntetos_boylan_regimes.png` y `fig02_master_thesis_results_panel.png` exportados a 300 DPI.

---

## Fase 6: Solución de Problemas Frecuentes para Principiantes (*Troubleshooting*)

### Problema 1: Al reiniciar o reconectar el Notebook, Python dice `ModuleNotFoundError: No module named 'lightgbm'` (o `'chronos'`)
* **¿Por qué ocurre?** En **Databricks Serverless Compute**, cada vez que la sesión entra en reposo por inactividad (aprox. 30–60 minutos sin uso), la máquina efímera se recicla y vuelve a su imagen limpia original.
* **Solución rápida:** Crea una primera celda al inicio del notebook que estés usando y ejecuta:
  ```python
  %pip install -r ../requirements.txt
  dbutils.library.restartPython()
  ```
  El comando mágico `%pip install` instala los paquetes en el entorno virtual efímero del notebook y `dbutils.library.restartPython()` refresca el intérprete de Python para que los reconozca de inmediato.

### Problema 2: Error de memoria (`OutOfMemoryError` o `SparkException: Job aborted due to stage failure`) al procesar los 31,8 millones de registros o 100.000 series
* **¿Por qué ocurre?** En **Databricks Free Edition**, la memoria RAM del nodo Driver es limitada si intentas traer millones de filas de Spark a Pandas con `.toPandas()` en un solo bloque.
* **Solución paso a paso:**
  1. **Reducir el tamaño de lote (`batch_size`):** En `configs/experiment_config.yaml`, bajo `scaling_benchmark`, reduce `batch_size_cpu` de `32` a `16` o `8`.
  2. **Usar la variante más ligera de Chronos:** En `configs/experiment_config.yaml`, asegúrate de usar `amazon/chronos-bolt-tiny` (que pesa menos de 40 MB en RAM y es hasta $250\times$ más rápido que el Chronos-T5 original).
  3. **Aumentar las particiones de Spark:** Antes de ejecutar `applyInPandas`, reparticiona el DataFrame en PySpark para que cada tarea procese fragmentos pequeños:
     ```python
     df_silver = df_silver.repartition(64, "article_id", "sales_channel_id")
     ```

### Problema 3: Error `CatalogNotFoundException: Catalog 'workspace' not found`
* **¿Por qué ocurre?** Algunas cuentas de Databricks llaman a su catálogo predeterminado `main` en lugar de `workspace`.
* **Solución:** Nuestro módulo `src/fashion_tsfm/config.py` ya incluye la función `ensure_unity_catalog_infrastructure(spark, cfg)` que detecta automáticamente si tu catálogo se llama `workspace`, `main` o `hive_metastore` y hace el cambio de forma transparente. Si quieres fijarlo manualmente, solo cambia `catalog: "main"` en la línea 22 de `configs/experiment_config.yaml`.

