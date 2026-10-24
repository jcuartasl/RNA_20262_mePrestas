# Ficha Técnica del Conjunto de Datos: Lending Club Credit Risk

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia — Facultad de Minas  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  
**Rol:** Datos, objetivos (PD y LGD) y pérdida esperada (P1)  
**Fecha de elaboración:** 18 de octubre de 2026  

---

## 1. Identificación y Fuente

- **Nombre del conjunto:** Credit Risk Analysis Dataset (Lending Club)
- **Autor / Publicador en Kaggle:** Ranadeep Saha (`ranadeep/credit-risk-dataset`)
- **Enlace oficial:** [https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset)
- **Fecha y hora de descarga:** 17 de octubre de 2026, 8:27 PM (UTC-5)
- **Archivos fuente incluidos:**
  - `loan.csv` (421.31 MB sin comprimir, extraído de `archive.zip` de 101.23 MB)
  - `LCDataDictionary.xlsx` (Diccionario oficial de variables de Lending Club)

---

## 2. Dimensiones y Cobertura Temporal

- **Número total de registros (filas):** 887,379
- **Número total de variables (columnas):** 74
- **Periodo temporal cubierto (`issue_d`):**
  - **Fecha inicial:** Junio de 2007 (`Jun-2007`)
  - **Fecha final:** Diciembre de 2015 (`Dec-2015`)
- **Distribución anual de créditos otorgados:**

| Año (`issue_year`) | Número de Créditos | Proporción (%) |
| :---: | :---: | :---: |
| **2007** | 603 | 0.07 % |
| **2008** | 2,393 | 0.27 % |
| **2009** | 5,281 | 0.60 % |
| **2010** | 12,537 | 1.41 % |
| **2011** | 21,721 | 2.45 % |
| **2012** | 53,367 | 6.01 % |
| **2013** | 134,755 | 15.19 % |
| **2014** | 235,628 | 26.55 % |
| **2015** | 421,094 | 47.45 % |
| **Total** | **887,379** | **100.00 %** |

> **Nota metodológica sobre el crecimiento:** El volumen de créditos creció de forma exponencial entre 2012 y 2015. El año 2015 concentra casi la mitad de los datos (47.45%), lo cual es determinante para el diseño de la partición temporal y la consideración del sesgo de madurez.

---

## 3. Composición de la Variable Objetivo Original (`loan_status`)

| Categoría Original (`loan_status`) | Conteo de Registros | Proporción (%) | Mapeo Previsto según Guía |
| :--- | :---: | :---: | :---: |
| `Current` | 601,779 | 67.82 % | Excluido (`NA` - sin resolver) |
| `Fully Paid` | 207,723 | 23.41 % | `0` (Buen pagador confirmado) |
| `Charged Off` | 45,248 | 5.10 % | `1` (Crédito incobrable / default) |
| `Late (31-120 days)` | 11,591 | 1.31 % | `1` (Mal pagador / default) |
| `Issued` | 8,460 | 0.95 % | Excluido (`NA` - sin historia) |
| `In Grace Period` | 6,253 | 0.70 % | Excluido (`NA` - periodo de gracia) |
| `Late (16-30 days)` | 2,357 | 0.27 % | Excluido (`NA` - comportamiento incierto) |
| `Does not meet the credit policy. Status:Fully Paid` | 1,988 | 0.22 % | `0` (Buen pagador confirmado) |
| `Default` | 1,219 | 0.14 % | `1` (Impago confirmado) |
| `Does not meet the credit policy. Status:Charged Off` | 761 | 0.09 % | `1` (Incumplido no pagó) |
| **Total** | **887,379** | **100.00 %** | — |

### Síntesis para el modelado de PD:
- **Casos indeterminados / vigentes a excluir (`NA`):** 618,849 registros (~69.74%).
- **Población con desempeño resuelto disponible para entrenamiento/evaluación:** 268,530 registros (~30.26%).
  - **Buenos pagadores (`target_pd = 0`):** 209,711 registros (78.10% del universo resuelto).
  - **Incumplidos / Malos pagadores (`target_pd = 1`):** 58,819 registros (21.90% del universo resuelto).
  - **Tasa de default observada en resueltos:** **21.90%**.

---

## 4. Análisis de Calidad de Datos: Variables con Alto Porcentaje de Nulos

De las 74 columnas presentes, **21 columnas** presentan más del 50% de valores faltantes (`null_pct > 50%`). La mayoría corresponde a variables añadidas recientemente en la plataforma (créditos conjuntos o líneas de crédito revolvente detalladas):

| # | Columna | Registros Nulos | % Nulos | Razón / Significado según Diccionario |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `dti_joint` | 886,870 | 99.94 % | Ratio DTI para solicitantes conjuntos (casi todos son individuales). |
| 2 | `annual_inc_joint` | 886,868 | 99.94 % | Ingreso anual conjunto reportado. |
| 3 | `verification_status_joint` | 886,868 | 99.94 % | Estado de verificación de ingresos del co-solicitante. |
| 4 | `il_util` | 868,762 | 97.90 % | Ratio de saldo total a límite en cuentas a plazos activas. |
| 5 | `mths_since_rcnt_il` | 866,569 | 97.65 % | Meses desde la cuenta a plazos abierta más reciente. |
| 6 | `inq_fi` | 866,007 | 97.59 % | Consultas de finanzas personales. |
| 7 | `open_rv_24m` | 866,007 | 97.59 % | Operaciones rotativas abiertas en últimos 24 meses. |
| 8 | `open_acc_6m` | 866,007 | 97.59 % | Cuentas abiertas en últimos 6 meses. |
| 9 | `all_util` | 866,007 | 97.59 % | Utilización en todas las líneas de crédito. |
| 10 | `inq_last_12m` | 866,007 | 97.59 % | Consultas de crédito en últimos 12 meses. |
| 11 | `total_cu_tl` | 866,007 | 97.59 % | Número de operaciones de cooperativas de crédito. |
| 12 | `open_il_12m` | 866,007 | 97.59 % | Cuentas a plazos abiertas en últimos 12 meses. |
| 13 | `max_bal_bc` | 866,007 | 97.59 % | Saldo máximo actual debido en cuentas rotativas. |
| 14 | `open_il_6m` | 866,007 | 97.59 % | Cuentas a plazos abiertas en últimos 6 meses. |
| 15 | `open_il_24m` | 866,007 | 97.59 % | Cuentas a plazos abiertas en últimos 24 meses. |
| 16 | `open_rv_12m` | 866,007 | 97.59 % | Operaciones rotativas abiertas en últimos 12 meses. |
| 17 | `total_bal_il` | 866,007 | 97.59 % | Saldo total actual de todas las cuentas a plazos. |
| 18 | `desc` | 761,353 | 85.80 % | Descripción en texto libre proporcionada por el prestatario. |
| 19 | `mths_since_last_record` | 750,326 | 84.56 % | Meses desde el último registro público despectivo. |
| 20 | `mths_since_last_major_derog` | 665,676 | 75.02 % | Meses desde la calificación 90 días o peor más reciente. |
| 21 | `mths_since_last_delinq` | 454,312 | 51.20 % | Meses desde la última morosidad del prestatario. |

> **Recomendación para Fase 1:** Las columnas 1 a 18 presentan una tasa de nulos superior al 80% y carecen de poder discriminante estable temporalmente, por lo que deben eliminarse del espacio de características. En el caso de variables de morosidad (`mths_since_last_delinq`), su ausencia suele indicar que el usuario nunca ha estado en mora; se puede optar por imputación especial o exclusión fundamentada.

---

## 5. Resumen de Tipos de Datos Iniciales

- **Columnas numéricas (`float64` / `int64`):** 52 variables.
- **Columnas categóricas / texto (`object` / `string`):** 22 variables.
- **Gráfica de soporte:** [`registros_por_anio.png`](../../results/figures/registros_por_anio.png) generada y disponible para el reporte.
