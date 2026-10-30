# Riesgo de crédito con redes neuronales (PD y LGD)

Trabajo 2 RNA · ¿Me prestas?. Estima la probabilidad de incumplimiento (PD),
la pérdida dado incumplimiento (LGD) y la pérdida esperada.

## Enlaces
- Reporte técnico (blog): jcuartasl
- Aplicación web: <URL>
- Video promocional: <URL>

## Equipo
| Persona | Nombre | Módulo principal |
|---|---|---|
| P1 | Omar Chavez | Datos, objetivos, pérdida esperada |
| P2 | Sebastián Villegas | MLP, PD, baseline LGD |
| P3 | Juan David Robayo | Pérdidas, LGD, baseline PD |
| P4 | Cesar Peñuela | Evaluación, scorecard, interpretabilidad |
| P5 | Jonhatan Cuartas | Infraestructura, app, blog |

## Requisitos
- Python <versión 3.13.15>
- Sistema probado: Windows 11
- No requiere internet al ejecutar.

## Instalación desde powershell
```bash
git clone https://github.com/jcuartasl/RNA_20262_mePrestas
cd <ruta/del/repo/RNA_20262_mePrestas>
python -m venv .venv
source .venv/Scripts/activate      # Git Bash en Windows
# .venv\Scripts\Activate.ps1       # PowerShell
pip install -r requirements.txt

#Definición de variables de entorno que se leen en la app web
$env:ME_PRESTAS_REPO_URL="https://github.com/jcuartasl/RNA_20262_mePrestas"
$env:ME_PRESTAS_VIDEO_URL="por definir"
```

## Datos
-Fuente: Kaggle, Credit Risk Analysis (ranadeep). Fecha de descarga: 17 de octubre de 2026.
-Registros originales: 887.379 · Variables: 74 · Periodo de originación: junio de 2007 a diciembre de 2015 (ver docs/blog/-ficha_tecnica_dataset.md).
-Datos procesados incluidos en data/processed/ y particiones en data/splits/; se generan con python -m src.experiments.prepare_data.
No hace falta el archivo original (loan.csv, más de 400 MB): el repo trae la versión procesada. El diccionario de variables está en data/raw/LCDataDictionary.xlsx.

## Ejecución (un comando por experimento)
| Qué | Comando |
|---|---|
| Preparar datos | `python -m src.experiments.prepare_data --config configs/default.yaml` |
| Entrenar PD (red) | `python -m src.experiments.train_pd --config configs/pd_mlp.yaml` |
| Entrenar LGD (red) | `python -m src.experiments.train_lgd --config configs/lgd_mlp.yaml` |
| 30 semillas | `python -m src.experiments.run_seeds --model pd_mlp --n_seeds 30` |
| Scorecard | `python -m src.experiments.build_scorecard --config configs/scorecard.yaml` |
| Pérdida esperada | `python -m src.experiments.expected_loss --config configs/el.yaml` |
| Figuras | `python -m src.experiments.make_figures` |
| Demo (< 2 min) | `python -m src.experiments.demo` |
| App web (principal) | `python -m app.api` → http://127.0.0.1:8000 |
| App web (alternativa) | `streamlit run app/app.py` |

### Cambiar parámetros sin editar código
```bash
python -m src.experiments.train_pd --config configs/pd_mlp.yaml --override train.lr=0.001 model.hidden=[64,32] model.dropout=0.2
```

## Aplicación web: qué datos pide y por qué
- El formulario recibe 21 variables que el preprocesador exige y que un solicitante puede declarar 
  o consultar en buró **antes del desembolso** (caso de uso de pre-aprobación).
- Unidad: **USD**, la de entrenamiento (Lending Club 2007-2015). El equivalente en COP es solo informativo
  (`ME_PRESTAS_COP_PER_USD`, por defecto 3140).
- Fijas (no se preguntan): `funded_amnt = funded_amnt_inv = loan_amnt` (= EAD), `policy_code = 1`, `pymnt_plan = n`,
  `application_type = INDIVIDUAL`, `initial_list_status` (moda) y `earliest_cr_line` (en `preprocessor.pkl` todas sus categorías
  quedaron como «infrecuentes», por lo que no afecta ninguna predicción).
- Campos opcionales vacíos se completan con mediana/moda del entrenamiento y la respuesta los lista (`por_defecto`).
- Prueba de coherencia: `pytest tests/test_servicio.py`.

## Semillas y reproducibilidad
- Semilla global: 90 (en `configs/default.yaml`).
- Semillas de las 30 corridas: 0–29.
- Particiones guardadas en `data/splits/`.
- Modelos entrenados en `models/` (la app y las figuras no reentrenan).

## Estructura del repositorio
```
app/            API (api.py) y app Streamlit (app.py)
index.html      Interfaz web que consume la API
configs/        Configuración YAML por experimento (default.yaml se hereda)
data/           raw (solo diccionario), processed y splits
docs/           hipotesis.md, ia_log.md y entradas del blog (docs/blog)
models/         Modelos entrenados y preprocesador
notebooks/      Análisis exploratorio
results/        figures/ y tables/ (todo se regenera desde el código)
src/            data, models, search, scorecard, expected_loss, evaluation, interpret, experiments, utils
tests/          Pruebas con pytest
```

## Resultados principales
Prueba temporal (originaciones de 2015), leídos de `results/tables/p4_pd_eval_test_temporal_resumen.csv`:

| Modelo | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Regresión logística | 0,674 | 0,348 | 0,257 | 0,178 |
| Red neuronal (64, 32) | 0,683 | 0,367 | 0,268 | 0,175 |

Estas cifras corresponden a una semilla en prueba; las 30 semillas están en validación. Los resultados de LGD y pérdida esperada, y su discusión, están en el reporte técnico.
## Pruebas
`pytest tests/`

## Licencia y créditos
Datos: Lending Club vía Kaggle (ranadeep). Referencias en el reporte técnico.