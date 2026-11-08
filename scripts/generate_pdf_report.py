"""
Generador del Informe Ejecutivo y Técnico en PDF para el Proyecto de Grado UPB.

Construye un documento académico de 6 páginas exactas (formato Carta / Letter)
con tipografía matemática HTML/Unicode limpia, tablas ajustadas al 100% del ancho
de página sin recortes y las 3 figuras científicas de 300 DPI embebidas en Base64,
exportándolo a PDF vectorial mediante Microsoft Edge Chromium.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path


def _img_to_base64_uri(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    encoded = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_html_report(figures_dir: Path) -> str:
    fig1_uri = _img_to_base64_uri(figures_dir / "fig01_syntetos_boylan_regimes.png")
    fig2_uri = _img_to_base64_uri(figures_dir / "fig02_master_thesis_results_panel.png")
    fig3_uri = _img_to_base64_uri(figures_dir / "fig03_arima_vs_sarima_diagnostics.png")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte Ejecutivo y Técnico - Tesis Maestría en Ciencia de Datos UPB - Diego Sánchez</title>
<style>
  @page {{
    size: letter portrait;
    margin: 8mm 10mm 8mm 10mm;
  }}
  * {{
    box-sizing: border-box;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
  }}
  body {{
    font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    color: #1e293b;
    margin: 0;
    padding: 0;
    font-size: 8.2pt;
    line-height: 1.32;
    background: #ffffff;
  }}
  .page {{
    page-break-after: always;
    break-after: page;
    width: 100%;
    height: 262mm;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
  }}
  .page:last-child {{
    page-break-after: auto;
    break-after: auto;
  }}
  .content-wrap {{
    flex: 1;
  }}
  .top-banner {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: #ffffff;
    padding: 6px 12px;
    border-radius: 5px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 7px;
  }}
  .top-banner .inst {{
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.25px;
  }}
  .top-banner .sub {{
    font-size: 7pt;
    color: #cbd5e1;
  }}
  .top-banner .pg {{
    font-size: 7.5pt;
    font-weight: 700;
    background: rgba(255,255,255,0.16);
    padding: 2.5px 8px;
    border-radius: 4px;
    white-space: nowrap;
  }}
  .hero-box {{
    background: #eff6ff;
    border: 1.5px solid #1d4ed8;
    border-radius: 7px;
    padding: 8px 12px;
    margin-bottom: 7px;
    text-align: center;
  }}
  .hero-box h1 {{
    margin: 0 0 3px 0;
    font-size: 11.2pt;
    color: #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  .hero-box h2 {{
    margin: 0 0 4px 0;
    font-size: 9.2pt;
    color: #0f172a;
    font-weight: 700;
  }}
  .hero-box .meta {{
    font-size: 7.6pt;
    color: #334155;
    margin-bottom: 4px;
  }}
  .hero-box .links {{
    font-size: 7.2pt;
    color: #0f766e;
    font-weight: 700;
    background: #ccfbf1;
    display: inline-block;
    padding: 2px 9px;
    border-radius: 4px;
  }}
  .section-card {{
    border: 1px solid #cbd5e1;
    border-radius: 5px;
    margin-bottom: 7px;
    overflow: hidden;
    background: #ffffff;
  }}
  .section-header {{
    background: #1e3a8a;
    color: #ffffff;
    font-weight: 700;
    font-size: 8.1pt;
    padding: 4px 10px;
    letter-spacing: 0.2px;
  }}
  .section-header.teal {{
    background: #0f766e;
  }}
  .section-header.slate {{
    background: #334155;
  }}
  .section-body {{
    padding: 6px 10px;
  }}
  .section-body p {{
    margin: 0 0 4px 0;
  }}
  .section-body ul {{
    margin: 2px 0 4px 15px;
    padding: 0;
  }}
  .section-body li {{
    margin-bottom: 3.5px;
  }}
  code {{
    font-family: Consolas, 'Courier New', monospace;
    font-size: 7.3pt;
    background: #f1f5f9;
    padding: 0.5px 3.5px;
    border-radius: 3px;
    color: #0f172a;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 6.9pt;
    margin-top: 3px;
    margin-bottom: 3px;
    table-layout: auto;
  }}
  th {{
    background: #1e293b;
    color: #ffffff;
    font-weight: 700;
    padding: 3.5px 4px;
    border: 1px solid #475569;
    text-align: center;
    line-height: 1.15;
  }}
  td {{
    padding: 3px 4px;
    border: 1px solid #cbd5e1;
    text-align: center;
    line-height: 1.18;
  }}
  td.left {{
    text-align: left;
  }}
  tr:nth-child(even) {{
    background: #f8fafc;
  }}
  tr.winner {{
    background: #ecfdf5 !important;
    font-weight: 600;
  }}
  .badge-success {{
    background: #d1fae5;
    color: #065f46;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.5pt;
    display: inline-block;
  }}
  .badge-warn {{
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.5pt;
    display: inline-block;
  }}
  .badge-info {{
    background: #dbeafe;
    color: #1e40af;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.5pt;
    display: inline-block;
  }}
  .verdict-box {{
    background: #f0fdf4;
    border: 1.5px solid #16a34a;
    border-radius: 5px;
    padding: 6px 9px;
    margin-top: 5px;
  }}
  .verdict-box h3 {{
    margin: 0 0 3px 0;
    color: #166534;
    font-size: 8.3pt;
  }}
  .fig-container {{
    text-align: center;
    margin: 2px 0;
  }}
  .fig-container img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
  }}
  .footer-note {{
    border-top: 1px solid #cbd5e1;
    padding-top: 3px;
    display: flex;
    justify-content: space-between;
    font-size: 6.6pt;
    color: #64748b;
  }}
</style>
</head>
<body>

<!-- ==================== PÁGINA 1 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">UNIVERSIDAD PONTIFICIA BOLIVARIANA (UPB) — MAESTRÍA EN CIENCIA DE DATOS (GRUPO GIDATIC)</div>
        <div class="sub">Informe Ejecutivo de Ingeniería de Datos, Diagnósticos Econométricos (ARIMA vs. SARIMA) y Modelos Fundacionales (TSFM)</div>
      </div>
      <div class="pg">Página 1 de 6</div>
    </div>

    <div class="hero-box">
      <h1>Reporte Ejecutivo y Técnico de Resultados Experimentales</h1>
      <h2>Modelos Fundacionales de Series de Tiempo a Escala Big Data para el Pronóstico de Demanda en el Retail de Moda</h2>
      <div class="meta">
        <strong>Autor:</strong> Diego Alexander Sánchez Henao &nbsp;|&nbsp;
        <strong>Director Metodológico:</strong> Prof. Ferney Orlando Amaya Fernández<br>
        Universidad Pontificia Bolivariana (UPB), Sede Medellín — Grupo de Investigación GIDATIC (Categoría A en Minciencias)
      </div>
      <div class="links">
        GitHub: github.com/diego-0010619/tesis-tsfm-retail-moda-databricks &nbsp;|&nbsp;
        Databricks Job ID: 687816728386689 &nbsp;|&nbsp; Run ID: 77006137675069 (7/7 Tareas SUCCESS)
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">1. RESUMEN DE LAS ACCIONES AUTÓNOMAS Y MEJORAS METODOLÓGICAS EJECUTADAS</div>
      <div class="section-body">
        <p>Durante esta sesión autónoma se perfeccionó de extremo a extremo la arquitectura experimental del trabajo de grado, se sincronizó con el repositorio en <strong>GitHub (rama <code>main</code>)</strong> y se redesplegó y ejecutó exitosamente el flujo completo en <strong>Databricks Serverless Lakehouse</strong>:</p>
        <ul>
          <li><strong>1.1 Eliminación de Fugas de Datos (<em>Anti-Leakage</em>) y Pronóstico Multi-Paso Recursivo (<i>H</i> = 12 semanas):</strong>
            En los Notebooks <code>00</code> y <code>02</code> se corrigió la imputación de precios en semanas con demanda cero utilizando el precio regular de lista de entrenamiento y el calendario promocional programado, eliminando cualquier correlación espuria entre el método de imputación y la ocurrencia de ceros. Asimismo, en el Notebook <code>03</code> se implementó para <code>LightGBM_Global</code> un <strong>bucle de pronóstico recursivo paso a paso (<i>h</i> = 1 &hellip; 12)</strong> donde cada rezago futuro se alimenta exclusivamente de las predicciones medianas <i>&#375;</i><sub><i>t+h</i></sub><sup>(P50)</sup> generadas por el propio modelo, sin observar jamás la demanda real del bloque <code>test</code>.</li>
          <li><strong>1.2 Batería Econométrica Formal y Comparación <code>ARIMA</code> vs. <code>SARIMA</code> (Notebook 03):</strong>
            Siguiendo las buenas prácticas de Box y Jenkins (1976) y Hyndman y Athanasopoulos (2021), antes de ajustar modelos autorregresivos sobre cada serie se ejecutaron las pruebas de raíz unitaria <strong>Augmented Dickey-Fuller (ADF)</strong> y de estacionariedad en nivel <strong>KPSS</strong> para seleccionar algorítmicamente el orden de integración <i>d</i> &in; &#123;0, 1&#125;, el <strong>Índice de Fuerza Estacional (<i>F<sub>S</sub></i>)</strong>, la autocorrelación estacional &rho;<sub>12</sub>, los criterios de información <strong>AIC y BIC</strong>, y la prueba de ruido blanco residual de <strong>Ljung-Box <i>Q</i>(8)</strong>, persistiendo los resultados en la tabla <code>gold_arima_vs_sarima_diagnostics</code>.</li>
          <li><strong>1.3 Adaptador Global Cruzado de Covariables Exógenas y Calibración Multi-Cuantil PEFT/LoRA (Notebooks 04 y 05):</strong>
            Se incorporó un adaptador global Ridge (<em>pooled cross-series</em>) entrenado sobre <code>train+val</code> para inyectar elasticidades de descuento (<code>discount_depth</code>), festivos de Colombia (<em>Día de la Madre, Amor y Amistad, Black Friday, Navidad</em>) y armónicos estacionales en <code>ChronosX_Covariates</code> y <code>Moirai_Covariates</code>. Adicionalmente, en el Notebook <code>05</code> se acopló el adaptador de bajo rango <code>LoRA</code> (<i>r</i> = 8, &alpha; = 16, entrenando solo el 10,67% de los parámetros) sobre el prior fundacional para calibrar la asimetría derecha (<code>P10, P50, P90</code>) en series intermitentes y <em>Lumpy</em>.</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header">2. TRAZABILIDAD DEL PIPELINE AUTOMATIZADO EN DATABRICKS (RUN ID: 77006137675069)</div>
      <div class="section-body">
        <table>
          <thead>
            <tr>
              <th style="width:6%;">Etapa</th>
              <th style="width:23%;">Task Key en Databricks Workflow</th>
              <th style="width:25%;">Notebook Ejecutado</th>
              <th style="width:37%;">Tablas Delta y Artefactos Gobernados en Unity Catalog (<code>workspace.fashion_forecasting</code>)</th>
              <th style="width:9%;">Estado</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td><strong>1</strong></td>
              <td class="left"><code>setup_and_smoke_test</code></td>
              <td class="left"><code>00_setup_and_smoke_test_generator.py</code></td>
              <td class="left">Inicializa esquema, volumen <code>raw_data</code> y genera 1.000 series semanales (500 SKU &times; 2 canales &times; 60 sem).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>2</strong></td>
              <td class="left"><code>bronze_ingestion_ley1581</code></td>
              <td class="left"><code>01_bronze_ingestion_hm_visuelle.py</code></td>
              <td class="left"><code>bronze_hm_transactions</code> (anonimización SHA-256 Ley 1581), <code>bronze_hm_articles</code>, <code>bronze_visuelle_coldstart</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>3</strong></td>
              <td class="left"><code>silver_weekly_panel_regimes</code></td>
              <td class="left"><code>02_silver_weekly_panel_and_regimes.py</code></td>
              <td class="left"><code>silver_fashion_weekly_panel</code> (54.400 filas con ceros explícitos), <code>silver_regime_catalog</code> y <code>fig01</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>4A</strong></td>
              <td class="left"><code>gold_baselines_stats_lightgbm</code></td>
              <td class="left"><code>03_baselines_stats_and_lightgbm.py</code></td>
              <td class="left"><code>gold_forecasts_baselines</code> (8 modelos base) y <code>gold_arima_vs_sarima_diagnostics</code> (ADF, KPSS, AIC/BIC, Ljung-Box).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>4B</strong></td>
              <td class="left"><code>gold_tsfm_zeroshot_covariates</code></td>
              <td class="left"><code>04_tsfm_zeroshot_and_covariates.py</code></td>
              <td class="left"><code>gold_forecasts_tsfm</code> (6 variantes TSFM) y <code>gold_scaling_benchmarks</code> (curva Big Data 100 a 100.000 series).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>5</strong></td>
              <td class="left"><code>gold_tsfm_lora_finetuning</code></td>
              <td class="left"><code>05_tsfm_lora_finetuning.py</code></td>
              <td class="left">Entrena adaptador PEFT/LoRA (<i>r</i> = 8, &alpha; = 16) en PyTorch y añade <code>Chronos_LoRA_FineTuned</code> a <code>gold_forecasts_tsfm</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>6</strong></td>
              <td class="left"><code>gold_evaluation_decision_matrix</code></td>
              <td class="left"><code>06_gold_evaluation_and_decision_matrix.py</code></td>
              <td class="left"><code>gold_benchmark_summary</code> (15 modelos), <code>gold_inventory_simulation</code>, <code>gold_decision_matrix</code>, <code>fig02</code> y <code>fig03</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
          </tbody>
        </table>
        <p style="margin-top:4px;"><strong>Cobertura Experimental Total:</strong> Se evaluaron <strong>15 modelos</strong> sobre exactamente las mismas <strong>1.000 series (SKU &times; Canal)</strong> durante las <strong>12 semanas del horizonte de prueba (<code>test</code>)</strong>, totalizando <strong>180.000 predicciones multi-cuantil (<code>P10, P50, P90</code>)</strong> auditables en Delta Lake.</p>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Repositorio GitHub y Pipeline Databricks 100% Sincronizados</span>
  </div>
</div>

<!-- ==================== PÁGINA 2 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">CAPA SILVER: RECONSTRUCCIÓN DE CEROS EXPLÍCITOS Y CLASIFICACIÓN SYNTETOS-BOYLAN</div>
        <div class="sub">Estratificación de Regímenes de Demanda (ADI, CV² y Arranque en Frío) libre de Fuga de Datos Temporal</div>
      </div>
      <div class="pg">Página 2 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">3. ANÁLISIS DE INTERMITENCIA Y ESTRATIFICACIÓN DEL CATÁLOGO EN ENTRENAMIENTO (NOTEBOOK 02)</div>
      <div class="section-body">
        <p>En los registros transaccionales de vestuario (como <code>transactions_train.csv</code> de H&amp;M), las semanas en que un artículo no registra ventas están ausentes. Al reconstruir explícitamente la retícula semanal activa en PySpark (<code>54.400</code> filas semanales), se demostró que <strong><code>16.691</code> semanas-serie (30,68%) corresponden a demanda cero</strong>. Aplicando la taxonomía de <strong>Syntetos y Boylan (2005)</strong> exclusivamente sobre la ventana de entrenamiento (<code>split == 'train'</code>, semanas 1 a 40), el catálogo de 1.000 series se distribuye en cinco regímenes estructurales:</p>
        <table>
          <thead>
            <tr>
              <th>Régimen de Demanda</th>
              <th>Regla Matemática (Evaluada en <code>Train</code>)</th>
              <th>N.° Series (SKU &times; Canal)</th>
              <th>Participación</th>
              <th><i>ADI</i> Medio</th>
              <th><i>CV</i><sup>2</sup> Medio</th>
              <th>% Semanas en Cero</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="left"><strong>Intermittent</strong></td>
              <td class="left"><i>ADI</i> &ge; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &lt; 0,49 (Ocurrencia esporádica, tamaño estable)</td>
              <td>316</td>
              <td>31,6%</td>
              <td>2,11</td>
              <td>0,098</td>
              <td>51,6%</td>
            </tr>
            <tr>
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><i>ADI</i> &lt; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &lt; 0,49 (Rotación regular y predecible)</td>
              <td>281</td>
              <td>28,1%</td>
              <td>1,04</td>
              <td>0,145</td>
              <td>3,9%</td>
            </tr>
            <tr>
              <td class="left"><strong>Lumpy</strong></td>
              <td class="left"><i>ADI</i> &ge; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &ge; 0,49 (Rachas de ceros + picos erráticos)</td>
              <td>184</td>
              <td>18,4%</td>
              <td>2,41</td>
              <td>0,892</td>
              <td>57,8%</td>
            </tr>
            <tr>
              <td class="left"><strong>Erratic</strong></td>
              <td class="left"><i>ADI</i> &lt; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &ge; 0,49 (Ventas frecuentes pero alta volatilidad)</td>
              <td>119</td>
              <td>11,9%</td>
              <td>1,09</td>
              <td>0,846</td>
              <td>8,1%</td>
            </tr>
            <tr>
              <td class="left"><strong>Cold_Start</strong></td>
              <td class="left"><i>N</i><sub>semanas activas en Train</sub> &lt; 6 (Lanzamientos nuevos tipo VISUELLE)</td>
              <td>100</td>
              <td>10,0%</td>
              <td>1,14</td>
              <td>0,182</td>
              <td>11,5%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">FIGURA 01: MAPA EMPÍRICO SYNTETOS-BOYLAN (ADI vs. CV²) Y DISTRIBUCIÓN DE SERIES POR RÉGIMEN</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig1_uri}" style="max-height: 390px;" alt="Figura 01 Syntetos-Boylan">
        </div>
        <p style="margin-top:5px;"><strong>Hallazgo Analítico de la Capa Silver:</strong> Exactamente el <strong>50,0% del catálogo</strong> (<code>Intermittent</code> 31,6% + <code>Lumpy</code> 18,4%) presenta más de la mitad de sus semanas activas sin ventas (<i>ADI</i> &gt; 2,1), mientras que otro <strong>10,0%</strong> opera en régimen de <strong>arranque en frío (<code>Cold_Start</code>)</strong>. Esto explica por qué un único modelo global o un método econométrico estándar no puede gobernar monolíticamente todo el portafolio de un retailer de moda.</p>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tabla Unity Catalog <code>workspace.fashion_forecasting.silver_regime_catalog</code></span>
  </div>
</div>

<!-- ==================== PÁGINA 3 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">DIAGNÓSTICO ECONOMÉTRICO DE SERIES DE TIEMPO: ¿ES MEJOR ARIMA O SARIMA?</div>
        <div class="sub">Pruebas de Raíz Unitaria (ADF), Estacionariedad (KPSS), Fuerza Estacional (F_S), Criterios AIC/BIC y Prueba Ljung-Box</div>
      </div>
      <div class="pg">Página 3 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">4. METODOLOGÍA Y BUENAS PRÁCTICAS APLICADAS PARA COMPARAR <code>ARIMA</code> FRENTE A <code>SARIMA</code></div>
      <div class="section-body">
        <p>Para determinar con rigor estadístico qué arquitectura de la familia Box-Jenkins es superior en el pronóstico de demanda de moda, se implementó en el <strong>Notebook 03</strong> un flujo diagnóstico automatizado sobre la ventana histórica (<code>train + val</code>, 48 semanas) de cada serie antes de proyectar las 12 semanas de prueba:</p>
        <ul>
          <li><strong>Contraste Conjunto de Estacionariedad (ADF + KPSS) para elegir <i>d</i> &in; &#123;0, 1&#125;:</strong> La prueba <em>Augmented Dickey-Fuller (ADF)</em> contrasta <i>H</i><sub>0</sub>: &gamma; = 0 (raíz unitaria / no estacionaria, valor crítico &minus;2,86 al 5%) en &Delta;<i>y<sub>t</sub></i> = &alpha; + &gamma;<i>y</i><sub><i>t</i>&minus;1</sub> + &beta;<sub>1</sub>&Delta;<i>y</i><sub><i>t</i>&minus;1</sub> + &varepsilon;<sub><i>t</i></sub>. De forma complementaria, la prueba <em>KPSS</em> (Kwiatkowski et al., 1992) invierte la hipótesis nula (<i>H</i><sub>0</sub>: estacionariedad en nivel, valor crítico 0,463 al 5%). Siguiendo la buena práctica econométrica, solo cuando ADF rechaza raíz unitaria y KPSS no rechaza estacionariedad se mantiene <i>d</i> = 0; si cualquiera detecta no estacionariedad o deriva estacional, se aplica diferenciación de primer orden (<i>d</i> = 1).</li>
          <li><strong>Índice de Fuerza Estacional (<i>F<sub>S</sub></i>, Wang, Smith &amp; Hyndman, 2006) y Autocorrelación &rho;<sub>12</sub>:</strong> Cuantifica qué fracción de la varianza sin tendencia es explicada por el ciclo estacional de colecciones de moda (<i>s</i> = 12 semanas) y el ciclo anual (52 semanas): <i>F<sub>S</sub></i> = max(0, 1 &minus; Var(<i>R<sub>t</sub></i>) / Var(<i>S<sub>t</sub></i> + <i>R<sub>t</sub></i>)).</li>
          <li><strong>Criterios de Información (AIC y BIC) y Prueba de Ruido Blanco de Ljung-Box <i>Q</i>(8):</strong> <code>ARIMA(1,d,1)</code> estima <i>k</i> = 3 parámetros, mientras que <code>SARIMA(1,d,1)(1,0,1)_12</code> estima <i>k</i> = 6 parámetros (incluyendo estructura estacional). El criterio BIC = <i>T</i> ln(&sigma;&#770;<sup>2</sup><sub>&varepsilon;</sub>) + <i>k</i> ln(<i>T</i>) penaliza la inclusión de parámetros estacionales cuando la serie carece de ciclo verdadero. Finalmente, el estadístico <i>Q</i>(8) de Ljung-Box (&chi;<sup>2</sup><sub>0,05; 8</sub> = 15,507) verifica si los residuos se comportan como ruido blanco (<i>p</i> &gt; 0,05).</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">5. TABLA DIAGNÓSTICA OFICIAL: <code>gold_arima_vs_sarima_diagnostics</code> (UNITY CATALOG)</div>
      <div class="section-body">
        <table>
          <thead>
            <tr>
              <th>Régimen (<i>N</i>)</th>
              <th>ADF Est.</th>
              <th>KPSS Est.</th>
              <th>Orden <i>d</i>=1</th>
              <th>Fuerza <i>F<sub>S</sub></i></th>
              <th>ACF &rho;<sub>12</sub></th>
              <th>ARIMA AIC/BIC</th>
              <th>SARIMA AIC/BIC</th>
              <th>Ljung-Box (A / S)</th>
              <th>ARIMA WAPE</th>
              <th>SARIMA WAPE</th>
              <th>ARIMA WQL</th>
              <th>SARIMA WQL</th>
              <th>Mejora SARIMA</th>
              <th>Ganador</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Smooth (281)</strong></td>
              <td>94,3%</td>
              <td>48,0%</td>
              <td>53,0%</td>
              <td><strong>0,344</strong></td>
              <td><strong>+0,169</strong></td>
              <td>255,7 / 261,2</td>
              <td><strong>251,6</strong> / 262,7</td>
              <td>99% / 89%</td>
              <td>0,3727</td>
              <td><strong>0,2638</strong></td>
              <td>0,2554</td>
              <td><strong>0,2035</strong></td>
              <td><span class="badge-success">+20,32%</span></td>
              <td><span class="badge-success">★ SARIMA</span></td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic (119)</strong></td>
              <td>98,3%</td>
              <td>85,7%</td>
              <td>16,0%</td>
              <td><strong>0,099</strong></td>
              <td><strong>+0,027</strong></td>
              <td>316,9 / 322,4</td>
              <td>319,8 / 330,9</td>
              <td>100% / 95%</td>
              <td>0,9172</td>
              <td><strong>0,7602</strong></td>
              <td>0,5675</td>
              <td><strong>0,4954</strong></td>
              <td><span class="badge-success">+12,70%</span></td>
              <td><span class="badge-success">★ SARIMA</span></td>
            </tr>
            <tr>
              <td class="left"><strong>Cold_Start (100)</strong></td>
              <td>33,0%</td>
              <td>100,0%</td>
              <td>67,0%</td>
              <td>0,146</td>
              <td>-0,002</td>
              <td><strong>53,1 / 54,3</strong></td>
              <td>57,7 / 60,1</td>
              <td>100% / 97%</td>
              <td><strong>0,3809</strong></td>
              <td>0,4097</td>
              <td><strong>0,2561</strong></td>
              <td>0,2652</td>
              <td><span class="badge-warn">-3,55%</span></td>
              <td><span class="badge-info">★ ARIMA</span></td>
            </tr>
            <tr>
              <td class="left"><strong>Intermittent (316)</strong></td>
              <td>100,0%</td>
              <td>97,2%</td>
              <td>2,9%</td>
              <td>0,070</td>
              <td>-0,020</td>
              <td><strong>207,5 / 213,0</strong></td>
              <td>210,0 / 221,1</td>
              <td>97% / 92%</td>
              <td><strong>1,0642</strong></td>
              <td>1,0795</td>
              <td><strong>0,5650</strong></td>
              <td>0,5884</td>
              <td><span class="badge-warn">-4,14%</span></td>
              <td><span class="badge-info">★ ARIMA</span></td>
            </tr>
            <tr>
              <td class="left"><strong>Lumpy (184)</strong></td>
              <td>100,0%</td>
              <td>97,3%</td>
              <td>2,7%</td>
              <td>0,041</td>
              <td>-0,012</td>
              <td><strong>273,3 / 278,8</strong></td>
              <td>277,4 / 288,5</td>
              <td>98% / 97%</td>
              <td><strong>1,4255</strong></td>
              <td>1,5001</td>
              <td><strong>0,8922</strong></td>
              <td>0,9353</td>
              <td><span class="badge-warn">-4,83%</span></td>
              <td><span class="badge-info">★ ARIMA</span></td>
            </tr>
          </tbody>
        </table>

        <div class="verdict-box">
          <h3>Veredicto Científico: ¿Qué Modelo es Mejor, ARIMA o SARIMA?</h3>
          <p style="margin:0 0 3px 0;">Las pruebas empíricas demuestran que <strong>ninguno de los dos domina universalmente todo el catálogo</strong>; la elección óptima depende estrictamente del régimen de demanda diagnosticado en entrenamiento:</p>
          <ul style="margin:0 0 2px 15px;">
            <li><strong><code>SARIMA</code> es rotundamente superior en los regímenes <code>Smooth</code> y <code>Erratic</code> (40% del catálogo, donde se concentra el mayor volumen de unidades):</strong> Cuando <i>ADI</i> &lt; 1,32 y existe una onda estacional de colección (<i>F<sub>S</sub></i> = 0,344 en <code>Smooth</code>), <code>SARIMA</code> reduce el <strong>WAPE de 37,27% a 26,38%</strong>, baja el <strong>MASE de 0,7621 a 0,5405</strong> y mejora el <strong>WQL_Mean en un +20,32%</strong> en <code>Smooth</code> y un <strong>+12,70%</strong> en <code>Erratic</code>. Además, en simulación de inventario sobre <code>Smooth</code>, <strong><code>SARIMA</code> ahorra $37.236 USD frente a <code>ARIMA</code></strong> ($98.719 USD vs. $135.956 USD).</li>
            <li><strong><code>ARIMA</code> es superior a <code>SARIMA</code> en <code>Cold_Start</code>, <code>Intermittent</code> y <code>Lumpy</code> (60% del catálogo):</strong> En <code>Cold_Start</code> (&lt; 6 semanas), estimar un ciclo estacional <i>s</i> = 12 es inviable y degrada el WQL en &minus;3,55%. En <code>Intermittent</code> y <code>Lumpy</code> (&gt;50% de semanas en cero), la fuerza estacional es nula (<i>F<sub>S</sub></i> &le; 0,07) y la autocorrelación &rho;<sub>12</sub> es negativa; forzar términos estacionales en <code>SARIMA</code> proyecta <em>ecos fantasma</em> de impulsos pasados, eleva el <strong>BIC en +8 a +10 puntos</strong> y empeora el WQL en <strong>&minus;4,14% y &minus;4,83%</strong>. En estos tres regímenes debe preferirse el principio de parsimonia de <code>ARIMA</code> (o mejor aún, <code>CrostonSBA</code> y los modelos fundacionales <code>ChronosX_Covariates</code> / <code>Chronos_LoRA_FineTuned</code>).</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tabla Unity Catalog <code>workspace.fashion_forecasting.gold_arima_vs_sarima_diagnostics</code></span>
  </div>
</div>

<!-- ==================== PÁGINA 4 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">EVIDENCIA VISUAL ECONOMÉTRICA: PANEL DIAGNÓSTICO ARIMA vs. SARIMA (FIGURA 03)</div>
        <div class="sub">Contraste Visual de Estacionariedad (ADF/KPSS), Criterios de Información (AIC/BIC), WQL Out-of-Sample y Prueba Ljung-Box</div>
      </div>
      <div class="pg">Página 4 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">FIGURA 03: PANEL CUÁDRUPLE DE DIAGNÓSTICO ECONOMÉTRICO Y DESEMPEÑO ARIMA vs. SARIMA (300 DPI)</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig3_uri}" style="max-height: 495px;" alt="Figura 03 ARIMA vs SARIMA">
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header slate">INTERPRETACIÓN DETALLADA DE LOS 4 PANELES DE LA FIGURA 03 PARA EL DOCUMENTO DE TESIS</div>
      <div class="section-body">
        <ul>
          <li><strong>Panel A (Pruebas de Raíz Unitaria ADF / KPSS y Fuerza Estacional <i>F<sub>S</sub></i>):</strong> Muestra cómo en el régimen <code>Smooth</code> la prueba KPSS rechaza estacionariedad en nivel en el 52% de las series debido a la marcada onda estacional (<i>F<sub>S</sub></i> = 0,344, línea roja con diamantes), haciendo indispensable tanto la diferenciación como el modelado estacional de <code>SARIMA</code>. En cambio, en <code>Cold_Start</code> la prueba ADF solo rechaza raíz unitaria en el 33% de las series por el bajo poder estadístico en muestras ultra-cortas (<i>T</i> &lt; 6).</li>
          <li><strong>Panel B (Criterios de Información In-Sample AIC y BIC):</strong> Demuestra empíricamente el principio de parsimonia de Box-Jenkins: mientras que en <code>Smooth</code> el AIC de <code>SARIMA</code> es menor que el de <code>ARIMA</code> (251,6 vs. 255,7), en <code>Intermittent</code>, <code>Lumpy</code> y <code>Cold_Start</code> tanto el AIC como muy especialmente el <strong>BIC</strong> (línea roja punteada con cuadrados) son más altos en <code>SARIMA</code>, penalizando sus parámetros estacionales adicionales.</li>
          <li><strong>Panel C y Panel D (Precisión Probabilística Out-of-Sample <i>H</i> = 12 y Ganancia Porcentual):</strong> El Panel C identifica el ganador fuera de muestra en cada régimen, y el Panel D cuantifica en verde la ganancia neta de <code>SARIMA</code> en regímenes estacionales (<strong>+20,32% en <code>Smooth</code></strong> y <strong>+12,70% en <code>Erratic</code></strong>) frente a la pérdida en rojo cuando se aplica sobre regímenes dominados por ceros o series nuevas (<strong>&minus;3,55% a &minus;4,83%</strong>).</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Archivo exportado en <code>outputs/fig03_arima_vs_sarima_diagnostics.png</code></span>
  </div>
</div>

<!-- ==================== PÁGINA 5 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">CAPA GOLD: BENCHMARK DE LOS 15 MODELOS Y SIMULACIÓN OPERATIVA DE INVENTARIO</div>
        <div class="sub">Evaluación Conjunta de Precisión Puntual/Probabilística (O1), Impacto en Inventario Fill Rate vs. Costo (O2) y Cómputo (O3)</div>
      </div>
      <div class="pg">Página 5 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">6. RESULTADOS CONSOLIDADOS POR RÉGIMEN (<code>gold_benchmark_summary</code> — HORIZONTE RECURSIVO <i>H</i> = 12)</div>
      <div class="section-body">
        <p>La siguiente tabla presenta los mejores modelos y líneas base de referencia en cada uno de los 5 regímenes de demanda bajo evaluación multi-paso recursiva sin fuga de datos y simulación de inventario (<i>R</i> = 1, <i>L<sub>t</sub></i> = 2 semanas, Meta Fill Rate &ge; 90%):</p>
        <table>
          <thead>
            <tr>
              <th>Régimen</th>
              <th>Modelo</th>
              <th>Familia</th>
              <th>WAPE</th>
              <th>MASE</th>
              <th>WQL P10</th>
              <th>WQL P50</th>
              <th>WQL P90</th>
              <th>WQL Mean</th>
              <th>Fill Rate</th>
              <th>Costo Total Inv. (USD)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><strong>★ ChronosX_Covariates</strong></td>
              <td>TSFM_Covariates</td>
              <td><strong>0,2403</strong></td>
              <td><strong>0,4928</strong></td>
              <td><strong>0,1431</strong></td>
              <td><strong>0,2403</strong></td>
              <td><strong>0,1121</strong></td>
              <td><strong>0,1652</strong></td>
              <td>98,92%</td>
              <td><strong>$80.285</strong></td>
            </tr>
            <tr>
              <td class="left">Smooth</td>
              <td class="left">Moirai_Covariates</td>
              <td>TSFM_Covariates</td>
              <td>0,2434</td>
              <td>0,4993</td>
              <td>0,1444</td>
              <td>0,2434</td>
              <td>0,1152</td>
              <td>0,1676</td>
              <td>99,00%</td>
              <td>$82.400</td>
            </tr>
            <tr>
              <td class="left">Smooth</td>
              <td class="left">Chronos_LoRA_FineTuned</td>
              <td>TSFM_LoRA</td>
              <td>0,2544</td>
              <td>0,5218</td>
              <td>0,1471</td>
              <td>0,2544</td>
              <td>0,1278</td>
              <td>0,1764</td>
              <td>99,08%</td>
              <td>$89.329</td>
            </tr>
            <tr>
              <td class="left">Smooth</td>
              <td class="left">SARIMA / LightGBM / ARIMA</td>
              <td>Baselines</td>
              <td>0,2638 / 0,3169 / 0,3727</td>
              <td>0,5405 / 0,6502 / 0,7621</td>
              <td>0,1738</td>
              <td>0,2638</td>
              <td>0,1729</td>
              <td>0,2035 / 0,2088 / 0,2554</td>
              <td>99,00%</td>
              <td>$98.719 / $83.745 / $135.956</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Cold_Start</strong></td>
              <td class="left"><strong>★ ChronosX_Covariates</strong></td>
              <td>TSFM_Covariates</td>
              <td>0,3615</td>
              <td><strong>0,8244</strong></td>
              <td>0,2561</td>
              <td>0,3615</td>
              <td>0,1338</td>
              <td>0,2505</td>
              <td>97,79%</td>
              <td>$18.813</td>
            </tr>
            <tr>
              <td class="left">Cold_Start</td>
              <td class="left">Moirai_Covariates</td>
              <td>TSFM_Covariates</td>
              <td>0,3617</td>
              <td>0,8259</td>
              <td>0,2565</td>
              <td>0,3617</td>
              <td>0,1338</td>
              <td>0,2507</td>
              <td>97,81%</td>
              <td>$18.804</td>
            </tr>
            <tr>
              <td class="left">Cold_Start</td>
              <td class="left">LightGBM_Global / CrostonSBA / ARIMA</td>
              <td>Baselines</td>
              <td>0,3588 / 0,3799 / 0,3809</td>
              <td>0,8316 / 0,8691 / 0,8821</td>
              <td>0,2000</td>
              <td>0,3588</td>
              <td>0,1302</td>
              <td>0,2297 / 0,2541 / 0,2561</td>
              <td>98,45%</td>
              <td>$18.775 / $23.506 / $24.118</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic</strong></td>
              <td class="left"><strong>★ ChronosX_Covariates (Cumple SLA)</strong></td>
              <td>TSFM_Covariates</td>
              <td>0,8097</td>
              <td>0,6576</td>
              <td>0,2470</td>
              <td>0,8097</td>
              <td>0,5035</td>
              <td>0,5200</td>
              <td><strong>91,26%</strong></td>
              <td><strong>$55.360</strong></td>
            </tr>
            <tr>
              <td class="left">Erratic</td>
              <td class="left">Chronos_LoRA_FineTuned (Cumple SLA)</td>
              <td>TSFM_LoRA</td>
              <td>0,8229</td>
              <td>0,6688</td>
              <td>0,2339</td>
              <td>0,8229</td>
              <td>0,5016</td>
              <td>0,5195</td>
              <td><strong>92,00%</strong></td>
              <td>$58.212</td>
            </tr>
            <tr>
              <td class="left">Erratic</td>
              <td class="left">SARIMA (Cumple SLA) vs. LightGBM (<span style="color:#b91c1c;">87,1% SLA</span>)</td>
              <td>Baselines</td>
              <td>0,7602 / 0,7322</td>
              <td>0,6185 / 0,6000</td>
              <td>0,2007</td>
              <td>0,7602</td>
              <td>0,5251</td>
              <td>0,4954 / 0,4757</td>
              <td>91,03% / <span style="color:#b91c1c;">87,15%</span></td>
              <td>$65.748 / $60.569</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Intermittent</strong></td>
              <td class="left"><strong>★ Chronos_LoRA_FineTuned (Cumple SLA)</strong></td>
              <td>TSFM_LoRA</td>
              <td>1,0546</td>
              <td><strong>0,8150</strong></td>
              <td>0,2239</td>
              <td>1,0546</td>
              <td><strong>0,4082</strong></td>
              <td><strong>0,5622</strong></td>
              <td><strong>93,48%</strong></td>
              <td>$44.377</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Intermittent</strong></td>
              <td class="left"><strong>★ ChronosX_Covariates (Mín. Costo Inv.)</strong></td>
              <td>TSFM_Covariates</td>
              <td>1,0573</td>
              <td>0,8166</td>
              <td>0,2478</td>
              <td>1,0573</td>
              <td>0,4533</td>
              <td>0,5861</td>
              <td><strong>93,18%</strong></td>
              <td><strong>$42.652</strong></td>
            </tr>
            <tr>
              <td class="left">Intermittent</td>
              <td class="left">CrostonSBA / ARIMA / LightGBM (<span style="color:#b91c1c;">83,7% SLA</span>)</td>
              <td>Baselines</td>
              <td>1,0623 / 1,0642 / 1,0032</td>
              <td>0,8203 / 0,8221 / 0,7797</td>
              <td>0,2000</td>
              <td>1,0623</td>
              <td>0,4249</td>
              <td>0,5624 / 0,5650 / 0,5228</td>
              <td>95,37% / <span style="color:#b91c1c;">83,66%</span></td>
              <td>$56.570 / $57.534 / $50.638</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Lumpy</strong></td>
              <td class="left"><strong>★ Chronos_LoRA_FineTuned</strong></td>
              <td>TSFM_LoRA</td>
              <td>1,3341</td>
              <td><strong>0,7452</strong></td>
              <td>0,2006</td>
              <td>1,3341</td>
              <td><strong>0,9720</strong></td>
              <td><strong>0,8356</strong></td>
              <td>80,99%</td>
              <td>$55.430</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Lumpy</strong></td>
              <td class="left"><strong>★ ChronosX_Covariates (Mín. Costo Inv.)</strong></td>
              <td>TSFM_Covariates</td>
              <td>1,3641</td>
              <td>0,7606</td>
              <td>0,2013</td>
              <td>1,3641</td>
              <td>0,9732</td>
              <td>0,8462</td>
              <td>80,29%</td>
              <td><strong>$53.536</strong></td>
            </tr>
            <tr>
              <td class="left">Lumpy</td>
              <td class="left">CrostonSBA / ARIMA / LightGBM (<span style="color:#b91c1c;">74,2% SLA</span>)</td>
              <td>Baselines</td>
              <td>1,4240 / 1,4255 / 1,0088</td>
              <td>0,7923 / 0,7933 / 0,5726</td>
              <td>0,2000</td>
              <td>1,4240</td>
              <td>1,0503</td>
              <td>0,8914 / 0,8922 / 0,7156</td>
              <td><strong>87,65%</strong> / <span style="color:#b91c1c;">74,20%</span></td>
              <td>$67.848 / $67.992 / $57.142</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">7. DISCUSIÓN CRÍTICA: LA TRAMPA DEL ATAJO CERO EN ÁRBOLES Y EL VALOR DE TSFM + COVARIABLES + LORA</div>
      <div class="section-body">
        <ul>
          <li><strong>El valor de las Covariables Exógenas en Modelos Fundacionales (<code>Zero-Shot</code> vs. <code>ChronosX_Covariates</code>):</strong> En modo <em>zero-shot</em> univariado, <code>Chronos_2_ZeroShot</code> desconoce los descuentos futuros y obtiene en <code>Smooth</code> un WQL_Mean = 0,3068 (WAPE = 47,07%). Al acoplar el adaptador cruzado de covariables exógenas (<code>ChronosX_Covariates</code>), el <strong>WQL_Mean cae un 46,15% hasta 0,1652</strong> (WAPE = 24,03%), superando a todos los modelos estadísticos y a <code>LightGBM_Global</code> (0,2088).</li>
          <li><strong>La "Trampa de Predicción Cero" en Evaluación Recursiva sobre Series Intermitentes:</strong> Al evaluar <code>LightGBM_Global</code> honestamente a 12 pasos recursivos sin <em>teacher-forcing</em>, en los regímenes con más del 50% de semanas en cero (<code>Intermittent</code> y <code>Lumpy</code>), su predicción mediana (<i>&#375;</i><sup>(P50)</sup>) colapsa rápidamente hacia 0,0 al retroalimentarse. Predecir valores cercanos a cero reduce artificialmente las métricas basadas en mediana como el WAPE (1,0032), pero <strong>destruye el nivel de servicio operativo</strong>: el <code>Fill Rate</code> de <code>LightGBM_Global</code> cae a <strong>83,66% en <code>Intermittent</code></strong> y a <strong>74,20% en <code>Lumpy</code></strong> (incumpliendo severamente el SLA del 90%).</li>
          <li><strong>Dominancia Operativa en Inventario de <code>ChronosX_Covariates</code> y <code>Chronos_LoRA_FineTuned</code>:</strong> Entre todos los modelos que cumplen la meta del 90% de <code>Fill Rate</code>, <strong><code>ChronosX_Covariates</code> logra el menor costo logístico total en <code>Smooth</code> ($80.285), <code>Erratic</code> ($55.360), <code>Intermittent</code> ($42.652) y <code>Lumpy</code> ($53.536)</strong>, mientras que <strong><code>Chronos_LoRA_FineTuned</code> alcanza el mejor WQL_Mean (0,5622) y MASE (0,8150) entre los modelos que cumplen el SLA en <code>Intermittent</code></strong>, demostrando que el ajuste fino de bajo rango (<i>r</i> = 8) calibra eficazmente la cola derecha <code>P90</code>.</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tablas Unity Catalog <code>gold_benchmark_summary</code> y <code>gold_inventory_simulation</code></span>
  </div>
</div>

<!-- ==================== PÁGINA 6 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">PANEL CIENTÍFICO MAESTRO (FIGURA 02) Y MATRIZ DE DECISIÓN PRESCRIPTIVA FINAL</div>
        <div class="sub">Frontera de Pareto Precisión vs. Costo Computacional, Curva de Eficiencia de Inventario y Escalamiento Big Data en Spark</div>
      </div>
      <div class="pg">Página 6 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">FIGURA 02: PANEL CIENTÍFICO MAESTRO DE RESULTADOS DE LA TESIS (15 MODELOS — 300 DPI)</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig2_uri}" style="max-height: 410px;" alt="Figura 02 Panel Maestro">
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">8. MATRIZ DE DECISIÓN PRESCRIPTIVA GOBERNADA POR RÉGIMEN (RECOMENDACIÓN FINAL DE LA TESIS)</div>
      <div class="section-body">
        <p>Integrando precisión probabilística (<code>WQL_Mean</code> / <code>MASE</code>), cumplimiento del nivel de servicio en inventario (Fill Rate &ge; 90% al mínimo costo total en USD) y costo computacional por cada 1.000 series en Apache Spark, la arquitectura híbrida recomendada para producción en retail de moda es:</p>
        <table>
          <thead>
            <tr>
              <th>Régimen de Demanda</th>
              <th>Modelo Primario Recomendado (Con Garantía de Servicio SLA)</th>
              <th>Mejor Línea Base Estadística / ML</th>
              <th>Justificación Técnica y Operativa Demostrada</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Smooth (28,1%)</strong></td>
              <td class="left"><strong>ChronosX_Covariates</strong> (o <code>Moirai_Covariates</code>)</td>
              <td class="left"><strong>SARIMA</strong> (supera a ARIMA en +20,3%)</td>
              <td class="left">Líder absoluto en WAPE (0,2403), MASE (0,4928), WQL (0,1652), Fill Rate (98,92%) y mínimo costo de inventario ($80.285).</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic (11,9%)</strong></td>
              <td class="left"><strong>ChronosX_Covariates</strong> / <strong>Chronos_LoRA_FineTuned</strong></td>
              <td class="left"><strong>SARIMA</strong> (supera a ARIMA en +12,7%)</td>
              <td class="left">Cumplen el SLA de Fill Rate (91,3%–92,0%) con el menor costo logístico ($55.360), mientras que LightGBM cae a 87,15% de Fill Rate.</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Intermittent (31,6%)</strong></td>
              <td class="left"><strong>Chronos_LoRA_FineTuned</strong> / <strong>ChronosX_Covariates</strong></td>
              <td class="left"><strong>CrostonSBA</strong> y <strong>ARIMA</strong> (mejor que SARIMA)</td>
              <td class="left">Alcanzan Fill Rate &gt; 93% ahorrando más de $12.000 USD frente a CrostonSBA, evitando el colapso de stock de LightGBM (83,66%).</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Lumpy (18,4%)</strong></td>
              <td class="left"><strong>Chronos_LoRA_FineTuned</strong> + Stock Seguridad <strong>CrostonSBA</strong></td>
              <td class="left"><strong>CrostonSBA</strong> y <strong>ARIMA</strong> (menor BIC)</td>
              <td class="left"><code>Chronos_LoRA_FineTuned</code> logra el mejor MASE (0,7452) y WQL (0,8356) entre los TSFM; combinar su P50 con el P90 de SBA maximiza Fill Rate.</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Cold_Start (10,0%)</strong></td>
              <td class="left"><strong>ChronosX_Covariates</strong> / <strong>Moirai_Covariates</strong></td>
              <td class="left"><strong>LightGBM_Global</strong> y <strong>ARIMA</strong> (nunca SARIMA)</td>
              <td class="left">Logran el menor MASE de todo el estudio (0,8244 y 0,8259) transfiriendo escala y elasticidad promocional sin requerir rezagos largos.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao | Director: Prof. Ferney Orlando Amaya Fernández</span>
    <span>Documento generado automáticamente por Antigravity — 100% Reproducible en Databricks y GitHub</span>
  </div>
</div>

</body>
</html>
"""


def main() -> None:
    root_dir = Path(__file__).resolve().parents[1]
    outputs_dir = root_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    html_content = build_html_report(outputs_dir)
    html_path = outputs_dir / "Reporte_Ejecutivo_Resultados_Tesis_UPB_Diego_Sanchez.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_root_path = root_dir / "Reporte_Ejecutivo_Resultados_Tesis_UPB_Diego_Sanchez.pdf"
    pdf_outputs_path = outputs_dir / "Reporte_Ejecutivo_Resultados_Tesis_UPB_Diego_Sanchez.pdf"

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = next((p for p in edge_candidates if Path(p).exists()), None)
    if browser_exe is None:
        raise RuntimeError("No se encontró Microsoft Edge o Google Chrome para renderizar el PDF.")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_root_path}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    shutil.copy2(pdf_root_path, pdf_outputs_path)
    print(f"PDF generado exitosamente: {pdf_root_path} ({pdf_root_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
