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

## Instalación
```bash
git clone <URL>
cd <repo>
python -m venv .venv
source .venv/Scripts/activate      # Git Bash en Windows
# .venv\Scripts\Activate.ps1       # PowerShell
pip install -r requirements.txt

$env:ME_PRESTAS_REPO_URL="https://github.com/jcuartasl/RNA_20262_mePrestas"
$env:ME_PRESTAS_REPORT_URL="por definir"
$env:ME_PRESTAS_VIDEO_URL="por definir"
```

## Datos
- Fuente: Kaggle, Credit Risk Analysis (ranadeep). Fecha de descarga: <fecha>.
- Registros: <n> · Variables: <n> · Periodo: <desde–hasta>.
- Datos procesados incluidos en `data/processed/`; cómo se generaron: `python -m src.experiments.prepare_data`.
- Si no tienes el archivo original, no hace falta: el repo trae la versión procesada.

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

### Logística de PD y red de LGD (en este orden)
| Qué | Comando | Escribe |
|---|---|---|
| Logística de PD: búsqueda L1/L2 (27 configuraciones) | `python -m src.models.pd_logreg` | `results/tables/pd_logreg_busqueda.csv`, `models/pd_logreg.joblib` |
| Recalibración de Platt (usa el modelo guardado, no reentrena) | `python -m src.models.pd_logreg --recalibrar` | `results/tables/pd_logreg_platt.csv` |
| Figuras de la distribución de la LGD (sección 3.3) | `python -m src.models.lgd_mlp` | `results/figures/lgd_ajuste_beta_normal.png`, `lgd_qq_beta_normal.png`, `lgd_varianza_por_decil.png` |
| LGD 1: búsqueda pareada (12 configuraciones × 3 pérdidas) | `python -m src.search.lgd_random_search --fase buscar` | `results/tables/lgd_busqueda.csv`, `configs/lgd_mlp.yaml` |
| LGD 2: 30 semillas de la configuración final | `python -m src.experiments.run_seeds --model lgd_mlp --n_seeds 30` | `models/lgd_mlp_seed{0..29}.pt` |
| LGD 3: 30 semillas de las otras pérdidas y sensibilidad a ε | `python -m src.search.lgd_random_search --fase semillas` | `results/tables/lgd_semillas_{mse,beta}.csv`, `lgd_sensibilidad_eps.csv` |
| LGD 4: resumen de semillas, distribución predicha y curva | `python -m src.search.lgd_random_search --fase resumir` | `results/tables/lgd_semillas_resumen.csv`, `results/figures/lgd_pred_vs_obs.png`, `lgd_curva_aprendizaje.png` |
| LGD 5: techo de la tanh, calibración por deciles y ReLU | `python -m src.search.lgd_random_search --fase diagnostico` | `results/tables/lgd_diagnostico_techo.csv`, `lgd_calibracion_deciles.csv`, `lgd_techo_relu.csv` |
| LGD 6: estabilidad con la partición temporal | `python -m src.search.lgd_random_search --fase temporal` | `results/tables/lgd_estabilidad_temporal.csv` |
| LGD 7: distribución predicha frente a la observada | `python -m src.search.lgd_random_search --fase distribucion` | `results/tables/lgd_distribucion_val.csv`, `lgd_cuantiles_val.csv` |
| Red de LGD con una semilla (la de la configuración) | `python -m src.experiments.train_lgd` | `models/lgd_mlp_seed90.pt` |

- La fase `resumir` lee los modelos del paso 2 y toma sus tiempos de `results/tables/experiment_log.csv`.
- Cambiar la pérdida, la regularización o la tasa de aprendizaje de la red de LGD sin editar código:
  ```bash
  python -m src.experiments.train_lgd --override loss.name=beta train.l2=1e-4 train.lr=0.001 model.hidden=[64,32]
  ```
  Las claves se escriben completas (`train.lr`, no `lr`): una clave incompleta se ignora sin aviso, y
  `loss=beta` da error porque `loss` es una sección.

## Aplicación web: qué datos pide y por qué
- El formulario se construye desde los modelos guardados (`GET /esquema`, módulo `app/servicio.py`): 21 variables que el
  preprocesador exige y que un solicitante puede declarar o consultar en buró **antes del desembolso** (caso de uso de pre-aprobación).
- Unidad: **USD**, la de entrenamiento (Lending Club 2007-2015). El equivalente en COP es solo informativo
  (`ME_PRESTAS_COP_PER_USD`, por defecto 4000).
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


## Resultados principales
(tabla corta de métricas finales y enlace al reporte)

## Pruebas
`pytest tests/`

## Uso de IA
Resumen y enlace a `docs/ia_log.md`.

## Licencia y créditos
Fuente de los datos y referencias principales.