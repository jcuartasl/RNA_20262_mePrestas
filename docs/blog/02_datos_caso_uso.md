# Sección 2: Datos, Caso de Uso y Prevención de Fuga de Información

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  

---

## 1. Descripción del Conjunto de Datos

El estudio se fundamenta en el conjunto de datos **Credit Risk Analysis (Lending Club)** publicado por Ranadeep Saha en Kaggle ([enlace al dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)).

- **Fecha y hora de descarga:** 17 de octubre de 2026, 8:27 PM (UTC-5), registrada en `data/raw/fecha_consulta.py`.
- **Dimensiones originales:** 887,379 registros y 74 variables.
- **Periodo temporal:** Préstamos originados entre **junio de 2007** y **diciembre de 2015**.
- **Diccionario oficial:** Se contrastó cada campo con `LCDataDictionary.xlsx` provisto en el repositorio de datos original.

---

## 2. Caso de Uso del Modelo

### Definición del Problema Operativo
El modelo está diseñado para un **originador de crédito de consumo (banco comercial o FinTech)** en la etapa estricta de **admisión y pre-aprobación**.

- **¿Quién usa el modelo?** El motor automatizado de riesgo de crédito y los analistas del comité de admisión.
- **¿Para qué decisión?** Decidir si una solicitud de crédito entrante debe ser **aprobada o rechazada**, y determinar el monto de provisión o reserva preventiva de capital requerida para cubrir la pérdida esperada.
- **¿En qué momento?** En el instante exacto de la solicitud del cliente, **antes de que se realice el desembolso** de los fondos y antes de fijar condiciones contractuales posteriores.
- **¿Con qué información disponible?** Exclusivamente con datos sociodemográficos declarados y verificables del solicitante (ingresos, tipo de vivienda, estabilidad laboral) junto con el historial crediticio consolidado en el buró a la fecha de consulta (indagaciones recientes, morosidades históricas, líneas abiertas, saldo y tasa de utilización rotativa).

---

## 3. Implicaciones Metodológicas y Prevención de Fuga de Información (*Data Leakage*)

La coherencia con el caso de uso de pre-aprobación exige una auditoría rigurosa sobre el espacio de características. Incluir variables que surgen *después* del desembolso o que son resultado del propio sistema de decisión introduce **fuga de información retrospectiva**, inflando artificialmente las métricas (AUC sospechosamente altos) y tornando al modelo inservible en producción.

### A. Exclusión del Scoring Propietario de Lending Club
Las variables `grade`, `sub_grade`, `int_rate` y `installment` son asignadas por Lending Club *después* de que su propio algoritmo interno evaluó al solicitante y fijó la prima de riesgo. Utilizarlas como predictores crearía un sesgo circular: la red neuronal se limitaría a aproximar la regla de asignación de Lending Club en lugar de aprender el riesgo intrínseco del cliente a partir de su historial. Por tanto, se excluyen de la matriz principal de pre-aprobación.

### B. Matriz de Auditoría y Exclusión Sistemática de Fuga

A continuación se detalla la clasificación formal de las variables excluidas (disponible en [`results/tables/leakage_report.csv`](../../results/tables/leakage_report.csv)):

| Categoría de Fuga | Variables Excluidas | Justificación Económica y Operativa |
| :--- | :--- | :--- |
| **Pagos acumulados posteriores** | `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Registran pagos realizados a lo largo de la vida del crédito. En la admisión el cliente no ha pagado un solo centavo; su inclusión generaría una fuga perfecta. |
| **Eventos post-default / cobranza** | `recoveries`, `collection_recovery_fee` | Ocurren meses después de declarar el crédito como incobrable. Se excluyen en PD; se retienen exclusivamente para formular la variable continua de LGD. |
| **Saldos dinámicos en curso** | `out_prncp`, `out_prncp_inv` | Al originar, el saldo pendiente es exactamente el monto desembolsado. Cualquier reducción revela la amortización futura del préstamo. |
| **Fechas y eventos posteriores** | `last_pymnt_d`, `last_pymnt_amnt`, `last_credit_pull_d`, `next_pymnt_d` | Eventos transaccionales y revisiones de cartera ocurridas durante la vigencia del crédito. No existen al momento de admitir al cliente. |
| **Identificadores y texto libre** | `id`, `member_id`, `url`, `desc`, `title`, `emp_title`, `zip_code` | Identificadores únicos o categorías de altísima cardinalidad que inducen sobreajuste memorístico y vulneran la privacidad. |
| **Scoring propio de Lending Club** | `grade`, `sub_grade`, `int_rate`, `installment` | Variables comerciales derivadas del modelo existente de la plataforma; excluidas en pre-aprobación estricta para evaluar riesgo intrínseco. |

---

## 4. Hipótesis Explícitas de Riesgo de Negocio

Previo al ajuste de las redes neuronales y modelos de referencia, se formularon las siguientes hipótesis basadas en teoría económica y riesgo crediticio bancario (registradas en [`docs/hipotesis.md`](../hipotesis.md)):

1. **Hipótesis 1 (Capacidad de pago y sobreendeudamiento - DTI):** A mayor relación deuda-ingreso (`dti`), mayor será la probabilidad de incumplimiento ($PD$), debido a la menor holgura financiera del solicitante ante choques de liquidez.
2. **Hipótesis 2 (Indagaciones recientes de liquidez):** Un alto número de consultas en el buró de crédito en los últimos 6 meses (`inq_last_6mths`) se asocia con un mayor riesgo de impago, reflejando búsqueda desesperada de liquidez en múltiples entidades.
3. **Hipótesis 3 (Estrés en líneas rotativas):** Niveles elevados de utilización en tarjetas de crédito (`revol_util > 80%`) incrementan significativamente la probabilidad de default frente a solicitantes con baja utilización.
4. **Hipótesis 4 (Estabilidad laboral):** Prestatarios con menor antigüedad en el empleo (`emp_length < 1 year`) presentan una tasa de default superior frente a solicitantes consolidados con más de 10 años de estabilidad laboral.
5. **Hipótesis 5 (Monto financiado y severidad):** Montos de crédito elevados (`funded_amnt`) no solo incrementan la exposición ($EAD$), sino que exhiben una menor tasa de recuperación post-default (mayor $LGD$).

*Nota:* Estas hipótesis se contrastarán cuantitativamente al final del trabajo frente a las importancias de permutación / valores SHAP obtenidos por P4.
