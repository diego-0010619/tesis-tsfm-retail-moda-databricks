"""
Generador del Informe Ejecutivo y Técnico en PDF para el Proyecto de Grado UPB.

Construye un documento académico de 6 páginas exactas (formato Carta / Letter)
con los resultados empíricos reales del dataset H&M (31.788.324 transacciones,
105.542 artículos y 1.371.980 clientes) ejecutado en Databricks Serverless
(Job ID: 687816728386689 | Run ID: 413135334993039), con las 3 figuras de 300 DPI
embebidas en Base64 y exportado a PDF vectorial mediante Microsoft Edge Chromium.
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
<title>Reporte Ejecutivo y Técnico (Datos Reales H&M) - Tesis Maestría en Ciencia de Datos UPB - Diego Sánchez</title>
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
    font-size: 8.1pt;
    line-height: 1.31;
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
    padding: 7px 12px;
    margin-bottom: 7px;
    text-align: center;
  }}
  .hero-box h1 {{
    margin: 0 0 3px 0;
    font-size: 11pt;
    color: #1e3a8a;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  .hero-box h2 {{
    margin: 0 0 4px 0;
    font-size: 9.1pt;
    color: #0f172a;
    font-weight: 700;
  }}
  .hero-box .meta {{
    font-size: 7.5pt;
    color: #334155;
    margin-bottom: 4px;
  }}
  .hero-box .links {{
    font-size: 7.1pt;
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
    font-size: 8pt;
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
    margin-bottom: 3.2px;
  }}
  code {{
    font-family: Consolas, 'Courier New', monospace;
    font-size: 7.2pt;
    background: #f1f5f9;
    padding: 0.5px 3.5px;
    border-radius: 3px;
    color: #0f172a;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 6.8pt;
    margin-top: 3px;
    margin-bottom: 3px;
    table-layout: auto;
  }}
  th {{
    background: #1e293b;
    color: #ffffff;
    font-weight: 700;
    padding: 3.2px 4px;
    border: 1px solid #475569;
    text-align: center;
    line-height: 1.15;
  }}
  td {{
    padding: 2.8px 4px;
    border: 1px solid #cbd5e1;
    text-align: center;
    line-height: 1.16;
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
    font-size: 6.4pt;
    display: inline-block;
  }}
  .badge-warn {{
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.4pt;
    display: inline-block;
  }}
  .badge-info {{
    background: #dbeafe;
    color: #1e40af;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 6.4pt;
    display: inline-block;
  }}
  .verdict-box {{
    background: #f0fdf4;
    border: 1.5px solid #16a34a;
    border-radius: 5px;
    padding: 5.5px 9px;
    margin-top: 5px;
  }}
  .verdict-box h3 {{
    margin: 0 0 3px 0;
    color: #166534;
    font-size: 8.2pt;
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
        <div class="sub">Informe Ejecutivo Empírico sobre el Dataset Oficial H&amp;M (31,78M Transacciones), ARIMA vs. SARIMA y Modelos Fundacionales</div>
      </div>
      <div class="pg">Página 1 de 6</div>
    </div>

    <div class="hero-box">
      <h1>Reporte Ejecutivo de Resultados Empíricos Reales (Dataset Oficial H&amp;M)</h1>
      <h2>Modelos Fundacionales de Series de Tiempo a Escala Big Data para el Pronóstico de Demanda en el Retail de Moda</h2>
      <div class="meta">
        <strong>Autor:</strong> Diego Alexander Sánchez Henao &nbsp;|&nbsp;
        <strong>Director Metodológico:</strong> Prof. Ferney Orlando Amaya Fernández<br>
        Universidad Pontificia Bolivariana (UPB), Sede Medellín — Grupo de Investigación GIDATIC (Categoría A en Minciencias)
      </div>
      <div class="links">
        GitHub: github.com/diego-0010619/tesis-tsfm-retail-moda-databricks &nbsp;|&nbsp;
        Databricks Job ID: 687816728386689 &nbsp;|&nbsp; Run ID: 413135334993039 (7/7 Tareas SUCCESS)
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">1. INGESTIÓN DEL DATASET OFICIAL DE H&amp;M EN LOCAL Y EN DATABRICKS LAKEHOUSE (33,31 MILLONES DE REGISTROS)</div>
      <div class="section-body">
        <p>En cumplimiento del objetivo central del proyecto de grado, se descargó e integró el <strong>conjunto de datos real y completo de <em>H&amp;M Personalized Fashion Recommendations</em></strong> tanto en la carpeta física local del proyecto (<code>data/hm/</code>, <code>938,88 MB</code>) como en el volumen gobernado de Unity Catalog en Databricks (<code>/Volumes/workspace/fashion_forecasting/raw_data/hm/</code>), desactivando el modo sintético (<code>use_smoke_test: false</code>) y ejecutando el pipeline Medallion de principio a fin:</p>
        <ul>
          <li><strong>1.1 Fuentes Primarias Reales Ingestadas y Anonimizadas (Ley 1581 de 2012):</strong>
            <ul>
              <li><code>transactions_train.parquet</code> (<code>738,41 MB</code>): <strong>31.788.324 transacciones reales de compra</strong> entre septiembre de 2018 y septiembre de 2020 por fecha (<code>t_dat</code>), artículo (<code>article_id</code>), canal (<code>sales_channel_id</code>: 1 = Tienda física, 2 = Online) y precio pagado, anonimizando el identificador del comprador con hash criptográfico irreversible <code>SHA-256</code> (<code>customer_id_hash</code>).</li>
              <li><code>articles.csv</code> y <code>articles.parquet</code> (<code>40,53 MB</code>): <strong>105.542 artículos reales de vestuario y calzado</strong> con metadatos jerárquicos de producto, departamento, grupo de prenda, color y sección.</li>
              <li><code>customers.parquet</code> (<code>159,94 MB</code>): <strong>1.371.980 clientes únicos anonimizados</strong> con SHA-256 + <code>bronze_visuelle_coldstart</code> (<strong>5.577 referencias</strong> de arranque en frío).</li>
            </ul>
          </li>
          <li><strong>1.2 Reconstrucción de Ceros Explícitos sin Fuga Temporal y Normalización Local por Parches (<code>ReVIN</code>):</strong>
            Sobre las <strong>60 semanas completas lunes–domingo</strong> (excluyendo la semana truncada final de 2 días en septiembre de 2020), se reconstruyó la retícula semanal activa con relleno explícito de ceros (<code>demand_units = 0.0</code>), descuento empírico real (<code>empirical_markdown</code> frente al precio base de entrenamiento sin fuga hacia el horizonte <code>test</code>) y normalización local por parches recientes (<code>recent_4</code>, <code>recent_8</code>, <code>recent_12</code>) para capturar el decaimiento de ciclo de vida de las prendas reales de H&amp;M.</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header">2. TRAZABILIDAD DEL PIPELINE BIG DATA EJECUTADO EN DATABRICKS (RUN ID: 413135334993039)</div>
      <div class="section-body">
        <table>
          <thead>
            <tr>
              <th style="width:6%;">Etapa</th>
              <th style="width:23%;">Task Key en Databricks Workflow</th>
              <th style="width:24%;">Notebook Ejecutado</th>
              <th style="width:38%;">Tablas Delta y Artefactos Gobernados en Unity Catalog (<code>workspace.fashion_forecasting</code>)</th>
              <th style="width:9%;">Estado</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td><strong>1</strong></td>
              <td class="left"><code>setup_and_smoke_test</code></td>
              <td class="left"><code>00_setup_and_smoke_test_generator.py</code></td>
              <td class="left">Verifica esquema <code>workspace.fashion_forecasting</code>, volumen <code>raw_data</code> y modo <code>use_smoke_test: false</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>2</strong></td>
              <td class="left"><code>bronze_ingestion_ley1581</code></td>
              <td class="left"><code>01_bronze_ingestion_hm_visuelle.py</code></td>
              <td class="left"><code>bronze_hm_transactions</code> (31.788.324 filas SHA-256), <code>bronze_hm_articles</code> (105.542), <code>bronze_hm_customers</code> (1.371.980).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>3</strong></td>
              <td class="left"><code>silver_weekly_panel_regimes</code></td>
              <td class="left"><code>02_silver_weekly_panel_and_regimes.py</code></td>
              <td class="left">Clasifica todo el catálogo activo de H&amp;M y extrae el panel estratificado de 1.000 series reales (<code>53.755</code> filas semanales) + <code>fig01</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>4A</strong></td>
              <td class="left"><code>gold_baselines_stats_lightgbm</code></td>
              <td class="left"><code>03_baselines_stats_and_lightgbm.py</code></td>
              <td class="left"><code>gold_forecasts_baselines</code> (8 modelos base recursivos) y <code>gold_arima_vs_sarima_diagnostics</code> (ADF, KPSS, AIC/BIC, Ljung-Box).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>4B</strong></td>
              <td class="left"><code>gold_tsfm_zeroshot_covariates</code></td>
              <td class="left"><code>04_tsfm_zeroshot_and_covariates.py</code></td>
              <td class="left"><code>gold_forecasts_tsfm</code> (6 variantes TSFM con ReVIN local y covariables) y <code>gold_scaling_benchmarks</code> (100 a 100.000 series).</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>5</strong></td>
              <td class="left"><code>gold_tsfm_lora_finetuning</code></td>
              <td class="left"><code>05_tsfm_lora_finetuning.py</code></td>
              <td class="left">Entrena adaptador PEFT/LoRA (<i>r</i> = 8, &alpha; = 16) en PyTorch sobre H&amp;M real y añade <code>Chronos_LoRA_FineTuned</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
            <tr class="winner">
              <td><strong>6</strong></td>
              <td class="left"><code>gold_evaluation_decision_matrix</code></td>
              <td class="left"><code>06_gold_evaluation_and_decision_matrix.py</code></td>
              <td class="left"><code>gold_benchmark_summary</code> (15 modelos &times; 5 regímenes), <code>gold_inventory_simulation</code>, <code>gold_decision_matrix</code>, <code>fig02</code> y <code>fig03</code>.</td>
              <td><span class="badge-success">SUCCESS</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Datos Reales H&amp;M (31,78M Transacciones) en Local (<code>data/hm/</code>) y Databricks (<code>Run ID: 413135334993039</code>)</span>
  </div>
</div>

<!-- ==================== PÁGINA 2 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">CAPA SILVER: TAXONOMÍA SYNTETOS-BOYLAN SOBRE EL CATÁLOGO REAL DE H&amp;M</div>
        <div class="sub">Estratificación de Regímenes de Demanda (ADI, CV² y Arranque en Frío) Evaluada en Entrenamiento (Semanas 1 a 40)</div>
      </div>
      <div class="pg">Página 2 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">3. CARACTERIZACIÓN DE LOS 5 REGÍMENES EN EL BENCHMARK ESTRATIFICADO REAL DE H&amp;M (NOTEBOOK 02)</div>
      <div class="section-body">
        <p>A partir de las <strong>31.788.324 transacciones reales de H&amp;M</strong>, la capa Silver reconstruyó las series semanales por artículo y canal (<code>article_id &times; sales_channel_id</code>), clasificó el universo de referencias mediante los umbrales canónicos de <strong>Syntetos y Boylan (2005)</strong> (<i>ADI</i> = 1,32 y <i>CV</i><sup>2</sup> = 0,49) calculados estrictamente sobre la ventana de entrenamiento (<code>split == 'train'</code>), y seleccionó una muestra estratificada representativa de <strong>1.000 series reales de H&amp;M (<code>53.755</code> observaciones semanales)</strong>:</p>
        <table>
          <thead>
            <tr>
              <th>Régimen de Demanda (H&amp;M Real)</th>
              <th>Regla Matemática (Evaluada en <code>Train</code>)</th>
              <th>N.° Series (SKU &times; Canal)</th>
              <th>Participación</th>
              <th><i>ADI</i> Medio Real</th>
              <th><i>CV</i><sup>2</sup> Medio Real</th>
              <th>% Semanas en Cero</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="left"><strong>Intermittent</strong></td>
              <td class="left"><i>ADI</i> &ge; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &lt; 0,49 (Ocurrencia esporádica, tamaño estable)</td>
              <td>320</td>
              <td>32,0%</td>
              <td>1,421</td>
              <td>0,366</td>
              <td>29,4%</td>
            </tr>
            <tr>
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><i>ADI</i> &lt; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &lt; 0,49 (Rotación regular en tienda física y online)</td>
              <td>280</td>
              <td>28,0%</td>
              <td>1,017</td>
              <td>0,369</td>
              <td>1,6%</td>
            </tr>
            <tr>
              <td class="left"><strong>Lumpy</strong></td>
              <td class="left"><i>ADI</i> &ge; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &ge; 0,49 (Rachas sin venta + picos promocionales)</td>
              <td>180</td>
              <td>18,0%</td>
              <td>1,333</td>
              <td>1,121</td>
              <td>25,0%</td>
            </tr>
            <tr>
              <td class="left"><strong>Erratic</strong></td>
              <td class="left"><i>ADI</i> &lt; 1,32 &nbsp;y&nbsp; <i>CV</i><sup>2</sup> &ge; 0,49 (Ventas continuas pero alta volatilidad de lote)</td>
              <td>120</td>
              <td>12,0%</td>
              <td>1,000</td>
              <td>1,247</td>
              <td>0,0%</td>
            </tr>
            <tr>
              <td class="left"><strong>Cold_Start</strong></td>
              <td class="left"><i>N</i><sub>semanas activas en Train</sub> &lt; 6 (Lanzamientos recientes en colección 2020)</td>
              <td>100</td>
              <td>10,0%</td>
              <td>1,000</td>
              <td>0,387</td>
              <td>0,0%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">FIGURA 01: MAPA EMPÍRICO SYNTETOS-BOYLAN (ADI vs. CV²) SOBRE DATOS REALES DE H&amp;M</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig1_uri}" style="max-height: 390px;" alt="Figura 01 Syntetos-Boylan H&M Real">
        </div>
        <p style="margin-top:5px;"><strong>Hallazgo Estructural en H&amp;M Real:</strong> A diferencia de los datos sintéticos estacionarios, las series reales de H&amp;M en los regímenes <code>Erratic</code> (<i>CV</i><sup>2</sup> medio = 1,247) y <code>Lumpy</code> (<i>CV</i><sup>2</sup> medio = 1,121) exhiben una dispersión cuadrática de tamaño de pedido más de 3 veces superior al umbral crítico de 0,49, producto de rebajas profundas de liquidación (<em>markdowns</em>) y cambios de colección entre temporadas.</p>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tabla Unity Catalog <code>workspace.fashion_forecasting.silver_regime_catalog</code> (Datos Reales H&amp;M)</span>
  </div>
</div>

<!-- ==================== PÁGINA 3 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">DIAGNÓSTICO ECONOMÉTRICO EN H&amp;M REAL: ¿ES MEJOR ARIMA O SARIMA?</div>
        <div class="sub">Pruebas de Raíz Unitaria (ADF), Estacionariedad (KPSS), No-Estacionariedad I(1), Criterios AIC/BIC y Prueba Ljung-Box</div>
      </div>
      <div class="pg">Página 3 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">4. DESCUBRIMIENTO ECONOMÉTRICO EN H&amp;M REAL: DOMINANCIA DE SERIES NO ESTACIONARIAS <i>I</i>(1) POR CICLO DE VIDA</div>
      <div class="section-body">
        <p>Al ejecutar la batería econométrica formal del <strong>Notebook 03</strong> sobre las 1.000 series reales de H&amp;M (ventana histórica <code>train + val</code> de 48 semanas), emergió un hallazgo fundamental que distingue a los datos reales de moda de cualquier simulación académica:</p>
        <ul>
          <li><strong>Más del 60% de las series reales de H&amp;M son No-Estacionarias de Orden <i>d</i> = 1 (<i>I</i>(1)):</strong> Al aplicar conjuntamente la prueba de raíz unitaria <strong>Augmented Dickey-Fuller (ADF)</strong> (<i>H</i><sub>0</sub>: raíz unitaria) y la prueba <strong>KPSS</strong> (<i>H</i><sub>0</sub>: estacionariedad en nivel), el criterio conjunto selecciona diferenciación de primer orden (<strong><i>d</i> = 1</strong>) en el <strong>56,07% de las series <code>Smooth</code>, 62,50% de <code>Erratic</code>, 65,94% de <code>Intermittent</code>, 67,78% de <code>Lumpy</code> y 68,00% de <code>Cold_Start</code></strong>. Esto ocurre porque las prendas reales de vestuario siguen una curva de ciclo de vida asimétrica (lanzamiento, madurez y agotamiento progresivo de tallas hacia el verano de 2020).</li>
          <li><strong>Efecto de la Amortiguación Local en Modelos Box-Jenkins (<code>ARIMA(1,d,1)</code> vs. <code>SARIMA(1,d,1)(1,0,1)_12</code>):</strong> En series reales <i>I</i>(1) con decaimiento de inventario, anclar el pronóstico en el nivel local reciente (últimas 8 semanas) evita proyectar el volumen inflado de temporadas pasadas y permite contrastar con rigor cuándo aporta valor el operador estacional de 12 semanas.</li>
        </ul>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">5. TABLA DIAGNÓSTICA OFICIAL EN H&amp;M REAL: <code>gold_arima_vs_sarima_diagnostics</code></div>
      <div class="section-body">
        <table>
          <thead>
            <tr>
              <th>Régimen (<i>N</i>)</th>
              <th>ADF Est.</th>
              <th>KPSS Est.</th>
              <th>Orden <i>d</i>=1</th>
              <th>Fuerza <i>F<sub>S</sub></i></th>
              <th>ARIMA AIC / BIC</th>
              <th>SARIMA AIC / BIC</th>
              <th>Ljung-Box (A / S)</th>
              <th>ARIMA WAPE</th>
              <th>SARIMA WAPE</th>
              <th>ARIMA MASE</th>
              <th>SARIMA MASE</th>
              <th>ARIMA WQL</th>
              <th>SARIMA WQL</th>
              <th>Costo Inv. (A vs. S)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Erratic (120)</strong></td>
              <td>55,8%</td>
              <td>54,2%</td>
              <td><strong>62,5%</strong></td>
              <td>0,061</td>
              <td>273,0 / 278,5</td>
              <td>277,9 / 289,0</td>
              <td>94% / 94%</td>
              <td>1,4711</td>
              <td><strong>1,4515</strong></td>
              <td>1,5744</td>
              <td><strong>1,5407</strong></td>
              <td>0,8833</td>
              <td><strong>0,8785</strong></td>
              <td>$88.239 vs. <strong>$85.421</strong></td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Smooth (280)</strong></td>
              <td>58,2%</td>
              <td>66,8%</td>
              <td><strong>56,1%</strong></td>
              <td>0,036</td>
              <td>214,4 / 220,0</td>
              <td>219,4 / 230,5</td>
              <td>93% / 91%</td>
              <td>1,0397</td>
              <td><strong>1,0264</strong></td>
              <td>1,4910</td>
              <td><strong>1,4588</strong></td>
              <td><strong>0,6793</strong></td>
              <td>0,6820</td>
              <td>$159.259 vs. <strong>$155.689</strong></td>
            </tr>
            <tr>
              <td class="left"><strong>Intermittent (320)</strong></td>
              <td>46,6%</td>
              <td>57,8%</td>
              <td><strong>65,9%</strong></td>
              <td>0,045</td>
              <td><strong>66,1 / 71,6</strong></td>
              <td>70,6 / 81,6</td>
              <td>91% / 88%</td>
              <td><strong>1,2024</strong></td>
              <td>1,2085</td>
              <td>1,3804</td>
              <td><strong>1,3688</strong></td>
              <td><strong>0,7521</strong></td>
              <td>0,7675</td>
              <td><strong>$33.968</strong> vs. $34.289</td>
            </tr>
            <tr>
              <td class="left"><strong>Lumpy (180)</strong></td>
              <td>54,4%</td>
              <td>51,7%</td>
              <td><strong>67,8%</strong></td>
              <td>0,050</td>
              <td><strong>137,3 / 142,9</strong></td>
              <td>142,4 / 153,5</td>
              <td>87% / 86%</td>
              <td><strong>1,5957</strong></td>
              <td>1,6041</td>
              <td>1,7949</td>
              <td><strong>1,7717</strong></td>
              <td><strong>0,9679</strong></td>
              <td>0,9969</td>
              <td><strong>$40.285</strong> vs. $41.002</td>
            </tr>
            <tr>
              <td class="left"><strong>Cold_Start (100)</strong></td>
              <td>34,0%</td>
              <td>91,0%</td>
              <td><strong>68,0%</strong></td>
              <td>0,107</td>
              <td><strong>69,1 / 70,5</strong></td>
              <td>74,2 / 77,2</td>
              <td>98% / 96%</td>
              <td><strong>2,4540</strong></td>
              <td>2,5633</td>
              <td>1,4423</td>
              <td><strong>1,4341</strong></td>
              <td><strong>1,4279</strong></td>
              <td>1,5757</td>
              <td><strong>$79.077</strong> vs. $86.344</td>
            </tr>
          </tbody>
        </table>

        <div class="verdict-box">
          <h3>Veredicto Empírico sobre Datos Reales de H&amp;M: ¿Cuándo usar SARIMA y cuándo ARIMA?</h3>
          <ul style="margin:0 0 2px 15px;">
            <li><strong>En regímenes continuos sin ceros (<code>Erratic</code> y <code>Smooth</code>, 40% del catálogo): <code>SARIMA</code> supera a <code>ARIMA</code> en precisión puntual (<code>WAPE</code> y <code>MASE</code>), menor sesgo y menor costo operativo de inventario.</strong> En <code>Erratic</code>, <code>SARIMA</code> gana en todas las métricas (<code>WAPE = 1,4515</code> vs. <code>1,4711</code>; <code>MASE = 1,5407</code> vs. <code>1,5744</code>; <code>WQL = 0,8785</code> vs. <code>0,8833</code>, ahorrando <strong>$2.818 USD</strong> en inventario). En <code>Smooth</code>, <code>SARIMA</code> logra mejor <code>WAPE</code> (<code>1,0264</code> vs. <code>1,0397</code>), mejor <code>MASE</code> (<code>1,4588</code> vs. <code>1,4910</code>), menor sesgo (<code>0,3355</code> vs. <code>0,3597</code>) y ahorra <strong>$3.570 USD</strong> en inventario frente a <code>ARIMA</code>, con un empate técnico en <code>WQL</code> (<code>0,6820</code> vs. <code>0,6793</code>).</li>
            <li><strong>En regímenes intermitentes o de lanzamiento reciente (<code>Intermittent</code>, <code>Lumpy</code> y <code>Cold_Start</code>, 60% del catálogo): <code>ARIMA</code> supera claramente a <code>SARIMA</code>.</strong> En presencia de rachas de ceros (<i>ADI</i> &ge; 1,32) o historiales cortos (&lt; 6 semanas), los parámetros estacionales adicionales de <code>SARIMA</code> elevan el <strong>BIC entre +6,6 y +10,6 puntos</strong>, degradan el <code>WQL_Mean</code> en <strong>&minus;2,05% (<code>Intermittent</code>), &minus;3,00% (<code>Lumpy</code>) y &minus;10,35% (<code>Cold_Start</code>)</strong>, e incrementan el costo de inventario en <strong>+$8.299 USD</strong> combinados.</li>
          </ul>
        </div>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tabla Unity Catalog <code>workspace.fashion_forecasting.gold_arima_vs_sarima_diagnostics</code> (H&amp;M Real)</span>
  </div>
</div>

<!-- ==================== PÁGINA 4 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">EVIDENCIA VISUAL ECONOMÉTRICA EN H&amp;M REAL: PANEL ARIMA vs. SARIMA (FIGURA 03)</div>
        <div class="sub">Pruebas ADF/KPSS, Criterios de Información AIC/BIC, WQL Out-of-Sample y Diagnóstico de Ruido Blanco Ljung-Box</div>
      </div>
      <div class="pg">Página 4 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">FIGURA 03: PANEL CUÁDRUPLE DE DIAGNÓSTICO ECONOMÉTRICO ARIMA vs. SARIMA EN H&amp;M REAL (300 DPI)</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig3_uri}" style="max-height: 495px;" alt="Figura 03 ARIMA vs SARIMA H&M Real">
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header slate">LECTURA TÉCNICA DE LA FIGURA 03 PARA LA SUSTENTACIÓN DE GRADO</div>
      <div class="section-body">
        <ul>
          <li><strong>Panel A (Estacionariedad ADF / KPSS en Prendas Reales de H&amp;M):</strong> Evidencia empíricamente que solo entre el 46,6% y el 58,2% de las series con historial completo rechazan raíz unitaria en la prueba ADF al 5%, confirmando que la mayoría del catálogo textil sigue trayectorias integradas <i>I</i>(1) gobernadas por el ciclo de vida de cada colección.</li>
          <li><strong>Panel B (Penalización por Complejidad AIC y BIC):</strong> Demuestra que en los 5 regímenes el criterio bayesiano <strong>BIC de <code>ARIMA(1,d,1)</code> (línea azul oscura) es sistemáticamente inferior en ~10 puntos al de <code>SARIMA(1,d,1)(1,0,1)_12</code> (línea roja)</strong>, debido a que la rápida rotación de colecciones en H&amp;M amortigua la autocorrelación estacional fija a 12 semanas.</li>
          <li><strong>Panel C y Panel D (Desempeño Out-of-Sample <i>H</i> = 12 y Ruido Blanco Ljung-Box):</strong> El Panel C y el Panel D muestran que <code>SARIMA</code> logra ventaja en WQL fuera de muestra en <code>Erratic</code> (+0,54%) y reduce el WAPE/MASE en <code>Smooth</code>, mientras que en <code>Intermittent</code> (&minus;2,05%), <code>Lumpy</code> (&minus;3,00%) y sobre todo en <code>Cold_Start</code> (&minus;10,35%) el principio de parsimonia favorece a <code>ARIMA</code>. Entre el 86,1% y el 96,0% de las series superan el test de Ljung-Box (<i>p</i> &gt; 0,05).</li>
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
        <div class="inst">CAPA GOLD: BENCHMARK DE LOS 15 MODELOS SOBRE EL DATASET REAL DE H&amp;M</div>
        <div class="sub">Contraste de Modelos Fundacionales (Zero-Shot, Covariables y LoRA) frente a LightGBM y Estadísticos Clásicos</div>
      </div>
      <div class="pg">Página 5 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">6. RESULTADOS EMPÍRICOS CONSOLIDADOS EN H&amp;M REAL (<code>gold_benchmark_summary</code> — <i>H</i> = 12 SEMANAS)</div>
      <div class="section-body">
        <p>La siguiente tabla reporta el desempeño real fuera de muestra (verano de 2020 de H&amp;M) de las arquitecturas líderes de cada familia en los 5 regímenes de demanda, evaluando precisión puntual (<code>WAPE</code>, <code>MASE</code>), precisión probabilística (<code>WQL_Mean</code>), nivel de servicio (<code>Fill Rate</code>) y costo total de inventario en USD:</p>
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
              <th>Sesgo</th>
              <th>Fill Rate</th>
              <th>Costo Inv. (USD)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><strong>★ LightGBM_Global (Mín. WQL/WAPE)</strong></td>
              <td>Global_ML</td>
              <td><strong>0,7850</strong></td>
              <td><strong>1,0699</strong></td>
              <td><strong>0,2392</strong></td>
              <td><strong>0,7850</strong></td>
              <td>0,7999</td>
              <td><strong>0,6080</strong></td>
              <td>-0,2176</td>
              <td>68,25%</td>
              <td>$152.726</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><strong>★ Chronos_LoRA_FineTuned (Mejor TSFM Puntual)</strong></td>
              <td>TSFM_LoRA</td>
              <td><strong>0,9717</strong></td>
              <td><strong>1,3068</strong></td>
              <td>0,6046</td>
              <td>0,9717</td>
              <td>0,7018</td>
              <td>0,7594</td>
              <td><strong>+0,1896</strong></td>
              <td>78,91%</td>
              <td>$150.905</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Smooth</strong></td>
              <td class="left"><strong>★ Chronos_2 / ChronosX / Moirai_Cov (Mín. Costo)</strong></td>
              <td>TSFM</td>
              <td>1,0118 / 0,9752</td>
              <td>1,3753 / 1,3193</td>
              <td>0,5693</td>
              <td>0,9752</td>
              <td>0,6918</td>
              <td>0,7576 / 0,7632</td>
              <td>+0,1923</td>
              <td>79,93%</td>
              <td><strong>$147.926 / $149.880</strong></td>
            </tr>
            <tr>
              <td class="left">Smooth</td>
              <td class="left">ARIMA / SARIMA / CrostonSBA</td>
              <td>Statistical</td>
              <td>1,0397 / 1,0264 / 1,0513</td>
              <td>1,4910 / 1,4588 / 1,4040</td>
              <td>0,3027</td>
              <td>1,0264</td>
              <td><strong>0,6782</strong></td>
              <td>0,6793 / 0,6820 / 0,6826</td>
              <td>+0,3355</td>
              <td><strong>83,29% / 84,58%</strong></td>
              <td>$159.259 / $155.689</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic</strong></td>
              <td class="left"><strong>★ LightGBM_Global (Mín. WQL/Costo)</strong></td>
              <td>Global_ML</td>
              <td><strong>0,7465</strong></td>
              <td><strong>0,8332</strong></td>
              <td><strong>0,2459</strong></td>
              <td><strong>0,7465</strong></td>
              <td><strong>0,5872</strong></td>
              <td><strong>0,5265</strong></td>
              <td><strong>+0,1177</strong></td>
              <td>85,09%</td>
              <td><strong>$61.355</strong></td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic</strong></td>
              <td class="left"><strong>★ Moirai_Covariates / Chronos_LoRA (Mejor TSFM)</strong></td>
              <td>TSFM_Cov/LoRA</td>
              <td><strong>1,3128 / 1,3181</strong></td>
              <td><strong>1,3745 / 1,3742</strong></td>
              <td>1,0901</td>
              <td>1,3128</td>
              <td><strong>0,6468</strong></td>
              <td>1,0201 / 1,0216</td>
              <td>+0,8781</td>
              <td>89,35% / 89,67%</td>
              <td><strong>$76.000 / $77.444</strong></td>
            </tr>
            <tr>
              <td class="left">Erratic</td>
              <td class="left">Chronos_2_ZeroShot / CrostonSBA / SARIMA (SLA &ge; 90%)</td>
              <td>TSFM / Stat</td>
              <td>1,4140 / 1,3789 / 1,4515</td>
              <td>1,4570 / 1,4275 / 1,5407</td>
              <td>0,3872</td>
              <td>1,3789</td>
              <td>0,6456</td>
              <td>1,0219 / 0,8204 / 0,8785</td>
              <td>+1,0252</td>
              <td><strong>90,67% / 94,51% / 92,16%</strong></td>
              <td>$77.242 / $83.486 / $85.421</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Intermittent</strong></td>
              <td class="left"><strong>★ CrostonSBA (Óptimo Servicio + Mín. Costo Inv.)</strong></td>
              <td>Statistical</td>
              <td>1,0495</td>
              <td>1,1518</td>
              <td>0,2761</td>
              <td>1,0495</td>
              <td><strong>0,7256</strong></td>
              <td><strong>0,6838</strong></td>
              <td><strong>+0,0290</strong></td>
              <td><strong>78,05%</strong></td>
              <td><strong>$29.045</strong></td>
            </tr>
            <tr>
              <td class="left">Intermittent</td>
              <td class="left">Chronos_2_ZeroShot / ChronosX_Cov /Chronos_LoRA</td>
              <td>TSFM</td>
              <td><strong>1,0788 / 1,0981 / 1,1003</strong></td>
              <td><strong>1,1892 / 1,1998 / 1,2033</strong></td>
              <td>0,3427</td>
              <td>1,0788</td>
              <td>0,9337</td>
              <td>0,7945 / 0,8585 / 0,7976</td>
              <td>-0,0673</td>
              <td>61,98% / 62,39%</td>
              <td>$33.208 / $35.742</td>
            </tr>
            <tr>
              <td class="left">Intermittent</td>
              <td class="left">LightGBM_Global (Sesgo -27%) vs. ARIMA / SARIMA</td>
              <td>ML / Stat</td>
              <td>0,7866 / 1,2024 / 1,2085</td>
              <td>0,7875 / 1,3804 / 1,3688</td>
              <td>0,1967</td>
              <td>0,7866</td>
              <td>0,8364</td>
              <td>0,6269 / 0,7521 / 0,7675</td>
              <td><span style="color:#b91c1c;">-0,2699</span> / +0,1709</td>
              <td><span style="color:#b91c1c;">64,29%</span> / 73,26%</td>
              <td>$35.047 / $33.968</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Lumpy</strong></td>
              <td class="left"><strong>★ LightGBM_Global / CrostonSBA</strong></td>
              <td>ML / Stat</td>
              <td><strong>0,8728</strong> / 1,3902</td>
              <td><strong>0,9852</strong> / 1,5121</td>
              <td><strong>0,3041</strong></td>
              <td><strong>0,8728</strong></td>
              <td><strong>0,7402</strong></td>
              <td><strong>0,6390</strong> / 0,8980</td>
              <td><strong>-0,0087</strong> / +0,5227</td>
              <td><strong>76,46% / 76,17%</strong></td>
              <td><strong>$31.509</strong> / $36.510</td>
            </tr>
            <tr>
              <td class="left">Lumpy</td>
              <td class="left">Chronos_2_ZeroShot / Chronos_LoRA / ChronosX_Cov</td>
              <td>TSFM</td>
              <td><strong>1,3899 / 1,4141 / 1,4145</strong></td>
              <td><strong>1,5085 / 1,5040 / 1,5076</strong></td>
              <td>0,5581</td>
              <td>1,3899</td>
              <td>0,9086</td>
              <td>0,9522 / 1,0280 / 1,0569</td>
              <td>+0,4382</td>
              <td>69,89% / 70,04%</td>
              <td><strong>$34.224</strong> / $37.136</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Cold_Start</strong></td>
              <td class="left"><strong>★ LightGBM_Global (Mín. WQL) + Moirai_Cov (SLA &ge; 90%)</strong></td>
              <td>ML / TSFM_Cov</td>
              <td><strong>1,1540 / 2,0852</strong></td>
              <td><strong>1,0624 / 1,2070</strong></td>
              <td><strong>0,4968</strong></td>
              <td><strong>1,1540</strong></td>
              <td><strong>0,8668 / 0,9403</strong></td>
              <td><strong>0,8392</strong> / 1,6145</td>
              <td>+0,6982 / +1,7082</td>
              <td>89,30% / <strong>90,56%</strong></td>
              <td><strong>$62.186 / $66.084</strong></td>
            </tr>
            <tr>
              <td class="left">Cold_Start</td>
              <td class="left">Chronos_LoRA / ChronosX_Cov vs. ARIMA / CrostonSBA</td>
              <td>TSFM vs. Stat</td>
              <td><strong>2,0937 / 2,1414</strong> vs. 2,4540 / 2,7168</td>
              <td><strong>1,2574 / 1,2983</strong> vs. 1,4423 / 1,2680</td>
              <td>0,6668</td>
              <td>2,0937</td>
              <td>0,9517</td>
              <td>1,6022 vs. 1,4279 / 1,6927</td>
              <td>+1,7330 vs. +2,1492</td>
              <td><strong>91,27% / 91,60%</strong></td>
              <td><strong>$66.694</strong> vs. $79.077 / $83.453</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">7. TRES HALLAZGOS CIENTÍFICOS CLAVE SOBRE EL DATASET REAL DE H&amp;M PARA LA DISCUSIÓN DE LA TESIS</div>
      <div class="section-body">
        <ul>
          <li><strong>1. En <code>Smooth</code> y <code>Erratic</code>, los Modelos Fundacionales con Covariables y LoRA superan a todos los modelos estadísticos en precisión puntual (<code>WAPE</code> y <code>MASE</code>) y logran el menor costo de inventario en <code>Smooth</code>:</strong> En <code>Smooth</code>, <code>Chronos_LoRA_FineTuned</code> (<code>WAPE = 0,9717</code>, <code>MASE = 1,3068</code>), <code>Moirai_Covariates</code> (<code>WAPE = 0,9718</code>) y <code>ChronosX_Covariates</code> (<code>WAPE = 0,9752</code>) superan a <code>SARIMA</code> (<code>1,0264</code>), <code>ARIMA</code> (<code>1,0397</code>), <code>CrostonSBA</code> (<code>1,0513</code>) y <code>AutoETS</code> (<code>1,0736</code>), reduciendo además el costo total de inventario hasta <strong>$147.926–$150.905 USD</strong> (más económico incluso que <code>LightGBM_Global</code> con $152.726 USD y que <code>ARIMA</code> con $159.259 USD).</li>
          <li><strong>2. En <code>Cold_Start</code> (&lt; 6 semanas de historia), los TSFM superan ampliamente a todos los métodos estadísticos y cumplen el SLA del 90% de <code>Fill Rate</code>:</strong> Mientras <code>ARIMA</code>, <code>SARIMA</code>, <code>CrostonSBA</code> y <code>AutoETS</code> sufren errores severos (<code>WAPE</code> entre <code>2,45</code> y <code>8,26</code> y costos entre <strong>$79.077 y $183.694 USD</strong>), <code>Moirai_Covariates</code> (<code>WAPE = 2,0852</code>, <code>MASE = 1,2070</code>, <code>Fill Rate = 90,56%</code>, <strong>$66.084 USD</strong>) y <code>Chronos_LoRA_FineTuned</code> (<code>WAPE = 2,0937</code>, <code>Fill Rate = 91,27%</code>) ahorran más de <strong>$12.900 USD</strong> frente a <code>ARIMA</code> y <strong>$17.300 USD</strong> frente a <code>CrostonSBA</code> cumpliendo la meta del 90% de nivel de servicio.</li>
          <li><strong>3. En <code>Intermittent</code>, el sesgo negativo recursivo de <code>LightGBM_Global</code> (&minus;26,99%) hunde su nivel de servicio a 64,29%, consagrando a <code>CrostonSBA</code> como campeón operativo:</strong> Aunque <code>LightGBM_Global</code> logra el menor WQL (0,6269), al sub-pronosticar sistemáticamente las rachas intermitentes genera quiebres masivos de stock ($35.047 USD). En cambio, <code>CrostonSBA</code> alcanza un sesgo casi nulo (+2,90%), eleva el <code>Fill Rate</code> en <strong>+13,76 puntos porcentuales (78,05%)</strong> y reduce el costo logístico al mínimo absoluto del régimen (<strong>$29.045 USD</strong>).</li>
        </ul>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao</span>
    <span>Fuente: Tablas Unity Catalog <code>gold_benchmark_summary</code> y <code>gold_inventory_simulation</code> (Datos Reales H&amp;M)</span>
  </div>
</div>

<!-- ==================== PÁGINA 6 ==================== -->
<div class="page">
  <div class="content-wrap">
    <div class="top-banner">
      <div>
        <div class="inst">PANEL CIENTÍFICO MAESTRO EN H&amp;M REAL (FIGURA 02) Y MATRIZ PRESCRIPTIVA FINAL</div>
        <div class="sub">Frontera de Pareto Precisión vs. Cómputo, Curva de Eficiencia Operativa de Inventario y Escalamiento Big Data en Spark</div>
      </div>
      <div class="pg">Página 6 de 6</div>
    </div>

    <div class="section-card">
      <div class="section-header">FIGURA 02: PANEL CIENTÍFICO MAESTRO DE RESULTADOS SOBRE DATOS REALES DE H&amp;M (15 MODELOS — 300 DPI)</div>
      <div class="section-body">
        <div class="fig-container">
          <img src="{fig2_uri}" style="max-height: 410px;" alt="Figura 02 Panel Maestro H&M Real">
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header teal">8. MATRIZ DE DECISIÓN PRESCRIPTIVA GOBERNADA POR RÉGIMEN (ARQUITECTURA HÍBRIDA RECOMENDADA)</div>
      <div class="section-body">
        <p>En lugar de desplegar un único modelo monolítico para las 105.542 referencias de H&amp;M, la evidencia empírica demuestra que la política óptima en un Lakehouse retail consiste en un <strong>enrutador algorítmico gobernado por el régimen Syntetos-Boylan</strong>:</p>
        <table>
          <thead>
            <tr>
              <th>Régimen en H&amp;M Real</th>
              <th>Campeón de Precisión (WQL / WAPE)</th>
              <th>Campeón Operativo de Inventario (Servicio / Costo)</th>
              <th>Justificación Prescriptiva Basada en Evidencia Real (Run ID: 413135334993039)</th>
            </tr>
          </thead>
          <tbody>
            <tr class="winner">
              <td class="left"><strong>Smooth (28,0%)</strong></td>
              <td class="left"><strong>LightGBM_Global</strong> (WQL 0,608) + <strong>Chronos_LoRA</strong> (WAPE 0,972)</td>
              <td class="left"><strong>Chronos_2 / ChronosX_Cov</strong> ($147.926) y <strong>SARIMA</strong> ($155.689)</td>
              <td class="left">Combinar la mediana de <code>LightGBM_Global</code> / <code>Chronos_LoRA</code> con el cuantil P90 de <code>ChronosX_Covariates</code> o <code>SARIMA</code> minimiza el costo total y eleva el Fill Rate sobre el 80%.</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Erratic (12,0%)</strong></td>
              <td class="left"><strong>LightGBM_Global</strong> (WQL 0,527) + <strong>Moirai_Covariates</strong> (WAPE 1,313)</td>
              <td class="left"><strong>Chronos_2_ZeroShot</strong> (SLA 90,7%, $77.242) y <strong>CrostonSBA</strong> (94,5%)</td>
              <td class="left"><code>LightGBM_Global</code> minimiza el error medio ($61.355), pero cuando se exige SLA &ge; 90%, <code>Chronos_2</code> / <code>Moirai_Covariates</code> logran el menor costo superando a <code>SARIMA</code> ($85.421).</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Intermittent (32,0%)</strong></td>
              <td class="left"><strong>CrostonSBA</strong> (WQL 0,684) y <strong>Chronos_2_ZeroShot</strong> (MASE 1,189)</td>
              <td class="left"><strong>CrostonSBA</strong> ($29.045 USD, Fill Rate 78,05%) + <strong>ARIMA</strong></td>
              <td class="left"><code>CrostonSBA</code> evita el colapso por sesgo negativo de <code>LightGBM_Global</code> (Fill Rate 64,29%) y ahorra $6.002 USD; entre los Box-Jenkins debe usarse <code>ARIMA</code> (nunca <code>SARIMA</code>).</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Lumpy (18,0%)</strong></td>
              <td class="left"><strong>LightGBM_Global</strong> (WQL 0,639) y <strong>Chronos_LoRA</strong> (MASE 1,504)</td>
              <td class="left"><strong>LightGBM_Global</strong> ($31.509, Fill Rate 76,46%) y <strong>Chronos_2</strong> ($34.224)</td>
              <td class="left"><code>LightGBM_Global</code> y <code>Chronos_2_ZeroShot</code> logran el menor costo total de inventario ($31.509 y $34.224 USD), superando ampliamente a <code>ARIMA</code> ($40.285) y <code>SARIMA</code> ($41.002).</td>
            </tr>
            <tr class="winner">
              <td class="left"><strong>Cold_Start (10,0%)</strong></td>
              <td class="left"><strong>LightGBM_Global</strong> (WQL 0,839) y <strong>Moirai_Covariates</strong> (MASE 1,207)</td>
              <td class="left"><strong>Moirai_Covariates / ChronosX_Cov</strong> (SLA &ge; 90,6%, $66.084 USD)</td>
              <td class="left">Para referencias nuevas (&lt; 6 sem), <code>Moirai_Covariates</code>, <code>Chronos_LoRA</code> y <code>ChronosX_Cov</code> cumplen el SLA &ge; 90% ahorrando más de $12.900 USD frente a <code>ARIMA</code> y <code>CrostonSBA</code>.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="footer-note">
    <span>Proyecto de Grado — Maestría en Ciencia de Datos (UPB Medellín) | Autor: Diego Alexander Sánchez Henao | Director: Prof. Ferney Orlando Amaya Fernández</span>
    <span>Documento generado automáticamente por Antigravity con Datos Reales de H&amp;M — 100% Reproducible en Databricks y GitHub</span>
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
