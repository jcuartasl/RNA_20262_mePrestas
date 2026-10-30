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