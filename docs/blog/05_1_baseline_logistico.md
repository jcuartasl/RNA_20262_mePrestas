# 5.1 Regresión logística de referencia para PD

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  
**Responsable:** P3 · **Revisión:** P2 · **Estado:** borrador para integración

> Esta sección presenta el modelo de referencia contra el que se compara la red de PD (sección 5.2): una
> regresión logística estimada por máxima verosimilitud, con una búsqueda de regularización L1 y L2 con
> presupuesto declarado. Usa la **partición temporal** de PD, la misma de la red: entrenamiento 2007–2013
> ($n = 163.541$, tasa de incumplimiento 19,28 %), validación 2014 ($n = 74.154$, 26,22 %) y prueba 2015
> ($n = 30.835$, 25,46 %), con 82 variables (sección 3.1). La numeración de tablas (5.1.x) es provisional
> hasta la integración del blog.

---

## 1. Modelo

La logística supone que el incumplimiento de cada crédito es una Bernoulli con probabilidad
$p_i = \sigma(b + \mathbf{w}^\top \mathbf{x}_i)$. Maximizar su verosimilitud equivale a minimizar la
entropía cruzada binaria (BCE), como se deriva en la sección 3.3 (apartado 2). A esa pérdida se le suma
una penalización sobre los coeficientes $\mathbf{w}$, L2 ($\lVert\mathbf{w}\rVert_2^2$) o L1
($\lVert\mathbf{w}\rVert_1$), cuyo efecto se explica en la sección 3.5.

Se usó `LogisticRegression` de scikit-learn, que parametriza la fuerza de la penalización con
$C$: un $C$ pequeño es una penalización fuerte, y $C = \infty$ es la logística sin penalizar. En la
configuración del proyecto se anota la equivalencia $\lambda = 1/(nC)$ con la pérdida promedio. La L2 se
ajustó con `lbfgs` y la L1 con `saga`, porque `lbfgs` no admite L1. `saga` hace pasos proximales, y por
eso la L1 deja coeficientes exactamente en cero.

## 2. Búsqueda y presupuesto

La Tabla 5.1.1 resume el espacio de búsqueda y el presupuesto.

| Elemento | Valor |
|---|---|
| Rejilla de $C$ | 13 valores en escala logarítmica entre $10^{-4}$ y $10^{2}$ |
| Penalización | L2 (`lbfgs`) y L1 (`saga`) |
| Referencia sin penalizar | $C = \infty$ |
| Configuraciones entrenadas | 27 (13 × 2 + 1), una vez cada una |
| Tiempo total | 12,6 min (L2: 0,4 min; L1: 12,2 min) |
| Parámetros | 83 (82 coeficientes y el intercepto) |
| Criterio de selección | menor BCE en la validación temporal (el mismo de la red de P2) |

**Tabla 5.1.1.** Espacio de búsqueda y presupuesto de la logística. *Fuente:* elaboración propia;
`configs/pd_logreg.yaml`, `src/models/pd_logreg.py` y `results/tables/pd_logreg_busqueda.csv`.

La regresión logística no depende de la semilla con `lbfgs`, así que se entrena una sola vez y no
necesita 30 semillas: la regla de las 30 semillas del enunciado es para las redes.

## 3. Resultados de la búsqueda

La Tabla 5.1.2 muestra una selección de las 27 configuraciones: las mejores con L2, la logística sin
penalizar y el camino de la L1.

| Penalización | $C$ | BCE validación | AUC validación | Brier | Coeficientes distintos de 0 |
|---|---|---|---|---|---|
| **L2 (elegida)** | **0,01** | **0,54219** | **0,68296** | **0,18024** | 81 |
| L2 | 0,316 | 0,54221 | 0,68279 | 0,18019 | 81 |
| Sin penalizar | $\infty$ | 0,54224 | 0,68278 | 0,18020 | 81 |
| L2 | $10^{-4}$ | 0,55074 | 0,68018 | 0,18461 | 81 |
| L1 | 0,1 | 0,54228 | 0,68280 | 0,18024 | 76 |
| L1 | 0,01 | 0,54312 | 0,68255 | 0,18078 | 34 |
| L1 | 0,001 | 0,55308 | 0,66919 | 0,18499 | 12 |
| L1 | $10^{-4}$ | 0,58960 | 0,50000 | 0,19825 | 0 |

**Tabla 5.1.2.** Filas seleccionadas de la búsqueda (validación temporal 2014). *Fuente:* elaboración
propia; `results/tables/pd_logreg_busqueda.csv` (las 27 filas).

- **La regularización casi no cambia el resultado.** Para todo $C \geq 0{,}003$ con L2, la BCE de
  validación está entre 0,54219 y 0,54236, y la logística sin penalizar queda a $5\times10^{-5}$ de la
  elegida. Es lo esperable con 163.541 créditos para 83 parámetros: los datos dominan la penalización.
  Las diferencias del orden de $10^{-5}$ ni siquiera son monótonas en $C$ (por ejemplo, $C = 1$ da
  0,54230, peor que sus vecinos). *Hipótesis:* están dentro de la tolerancia del optimizador
  ($10^{-4}$) y no deben leerse como una preferencia real por $C = 0{,}01$.
- **Aun así, la penalización resuelve un problema de identificación.** Las dos columnas de `term`
  (36 y 60 meses) suman 1 en cada crédito, igual que la columna implícita del intercepto. Además, hay tres
  columnas constantes en entrenamiento: `policy_code`, `earliest_cr_line_infrequent_sklearn` y
  `application_type_INDIVIDUAL`. Sin penalización, los coeficientes de esas columnas no son únicos: muchas
  combinaciones dan las mismas predicciones. La L2 elige la de menor norma, y se ve en el modelo guardado:
  las dos columnas que valen 1 en todo el entrenamiento se reparten el papel del intercepto con −0,163 cada una, y
  `policy_code` queda en 0. Por eso se reportan 81 coeficientes distintos de cero y no 82.
- **La L1 ofrece un modelo mucho más pequeño a bajo costo.** Con $C = 0{,}01$ conserva 34 de las 82
  variables y pierde solo 0,0004 de AUC. Con 12 variables pierde 0,014. Es una opción si se quisiera una
  scorecard con pocas variables, pero no se eligió porque el criterio prerregistrado es la BCE.
- **La L1 es mucho más lenta**: 12,2 minutos frente a 0,4 de toda la L2, porque `saga` es un método
  estocástico y necesita muchas pasadas sobre los datos para converger.

## 4. Calibración global y cambio de época

La logística estimada por máxima verosimilitud cumple una propiedad exacta. Si el intercepto no se
penaliza, la derivada de la pérdida respecto de $b$ es $\sum_i (p_i - y_i)$, y en el óptimo vale cero:
**la PD media predicha en entrenamiento es igual a la tasa de incumplimiento de entrenamiento**. El
modelo guardado lo cumple (Tabla 5.1.3). La diferencia es de $5\times10^{-5}$, compatible con la
tolerancia del optimizador.

| Partición | PD media predicha | Tasa observada | Diferencia |
|---|---|---|---|
| Entrenamiento (2007–2013) | 0,1928 | 0,1928 | 0,0000 |
| Validación (2014) | 0,2151 | 0,2622 | −0,0471 |
| Prueba (2015) | 0,2214 | 0,2546 | −0,0332 |

**Tabla 5.1.3.** PD media predicha por la logística elegida frente a la tasa de incumplimiento observada.
*Fuente:* elaboración propia; `models/pd_logreg.joblib` aplicado a la partición temporal de PD.

La logística **subestima la PD en 2014 y 2015**, pero no es un error del ajuste. El modelo reproduce la
tasa del periodo en que se entrenó, y la tasa observada sube en los años siguientes. Parte de esa subida
se debe a la censura por derecha de los créditos recientes (sección 3.1): de las cohortes de 2014 y 2015
se observan sobre todo los créditos que ya se resolvieron, y entre ellos pesan más los incumplimientos
tempranos. La red de P2 tiene el mismo problema, aunque más leve (PD media de 0,239 en validación; sección
5.3).

**Recalibración de Platt.** Un desplazamiento como este se corrige con una recalibración ajustada **solo
en validación**. La de Platt reestima un intercepto $a$ y una pendiente $b$ sobre el logit de la PD:

$$p' = \sigma\big(a + b\,\text{logit}(p)\big).$$

Es una logística de una sola variable, sin penalizar, ajustada por máxima verosimilitud en la validación
2014. Dio $a = 0{,}193$ y $b = 0{,}925$. La Tabla 5.1.4 compara la PD original y la recalibrada.

| Partición | PD | PD media | Tasa observada | BCE | Brier | AUC |
|---|---|---|---|---|---|---|
| Validación 2014 | original | 0,2151 | 0,2622 | 0,5422 | 0,1802 | 0,6830 |
| Validación 2014 | Platt | 0,2622 | 0,2622 | 0,5351 | 0,1779 | 0,6830 |
| Prueba 2015 | original | 0,2214 | 0,2546 | 0,5376 | 0,1781 | 0,6738 |
| Prueba 2015 | Platt | 0,2684 | 0,2546 | 0,5338 | 0,1771 | 0,6738 |

**Tabla 5.1.4.** PD original y recalibrada con Platt (ajustado en validación 2014). La BCE usa el recorte
de `log_loss` de scikit-learn (apartado 7). *Fuente:* elaboración propia;
`python -m src.models.pd_logreg --recalibrar`, `results/tables/pd_logreg_platt.csv`.

- **En prueba 2015 la calibración mejora.** El error de la PD media pasa de −0,033 a +0,014, y bajan la
  BCE (de 0,5376 a 0,5338) y el Brier (de 0,1781 a 0,1771).
- **La AUC no cambia.** Con $b > 0$ la transformación es creciente y no altera el orden de los créditos.
  La discriminación del modelo sigue siendo la misma; solo cambia el nivel de la PD.
- **En 2015 Platt se pasa.** Aprende el nivel de 2014 (26,2 %), y la tasa de 2015 es algo menor (25,5 %).
  Una recalibración fija en un año no corrige un nivel que sigue cambiando de un año a otro.
- **$b = 0{,}925 < 1$**: en 2014 la logística original era algo sobreconfiada (sus logits se dispersaban de
  más) y Platt los comprime un poco.
- **Las cifras de validación con Platt son optimistas**, porque se ajustó con esos mismos datos. Por la
  misma propiedad del intercepto de arriba, su PD media iguala exactamente la tasa de 2014. La cifra
  honesta es la de prueba.

La evaluación completa de la calibración (curvas de calibración) está en la sección 5.3.

## 5. Frente a la red de PD

La Tabla 5.1.5 compara la logística con la red de P2 en presupuesto, complejidad y métricas, con la PD
original (sin Platt), que es la que evaluó P4.

| | Logística (P3) | MLP [64, 32] ReLU (P2) |
|---|---|---|
| Configuraciones de búsqueda | 27 | 12 |
| Entrenamientos totales | 27 | 63 |
| Minutos | 12,6 | 69,4 |
| Parámetros entrenables | 83 | 7.425 |
| AUC validación | 0,6830 | 0,6900 ± 0,0025 (30 semillas) |
| AUC validación: peor y mejor semilla | — | 0,6818 y 0,6950 |
| AUC prueba 2015 | 0,6738 | 0,6834 |
| Gini prueba 2015 | 0,348 | 0,367 |
| KS prueba 2015 | 0,257 | 0,268 |
| Brier prueba 2015 | 0,1781 | 0,1751 |

**Tabla 5.1.5.** Logística de referencia frente a la red de PD. En prueba se evaluó una sola semilla de la
red. *Fuente:* elaboración propia con las tablas de P4: `results/tables/p4_comparacion_justa_pd.csv`.

- La red mejora a la logística en todas las métricas de prueba, pero por poco: **0,010 de AUC** y
  0,003 de Brier.
- **No todas las semillas de la red superan a la logística.** La peor semilla (0,6818) queda por debajo
  de la logística (0,6830) en validación, así que la ventaja de la red depende en parte de la
  inicialización.
- El costo de la red es 90 veces más parámetros y unas 5,5 veces más tiempo de búsqueda. La logística,
  además, se puede leer directamente como scorecard (sección 3.7).

## 6. Coeficientes

La Tabla 5.1.6 muestra los 10 coeficientes de mayor valor absoluto de la logística elegida.

| Variable | Coeficiente | Tipo |
|---|---|---|
| `purpose_small_business` | +0,554 | indicadora (0/1) |
| `term_ 36 months` | −0,535 | indicadora (0/1) |
| `funded_amnt` | +0,413 | numérica estandarizada |
| `emp_length_missing` | +0,387 | indicadora (0/1) |
| `term_ 60 months` | +0,372 | indicadora (0/1) |
| `annual_inc` | −0,351 | numérica estandarizada |
| `funded_amnt_inv` | −0,311 | numérica estandarizada |
| `purpose_credit_card` | −0,288 | indicadora (0/1) |
| `purpose_car` | −0,280 | indicadora (0/1) |
| `revol_util` | +0,247 | numérica estandarizada |

**Tabla 5.1.6.** Los 10 coeficientes de mayor valor absoluto de la logística elegida (un coeficiente
positivo sube el logit de la PD). Intercepto: −0,322. *Fuente:* elaboración propia;
`models/pd_logreg.joblib`.

Los coeficientes no deben leerse uno a uno como "efecto de la variable", por tres razones:

- **Escalas distintas.** Las numéricas están estandarizadas (el coeficiente es el cambio del logit por una
  desviación estándar) y las indicadoras no (el cambio es por pasar de 0 a 1). Sus magnitudes no son
  comparables entre sí.
- **Colinealidad.** `funded_amnt` (+0,41) y `funded_amnt_inv` (−0,31) son casi la misma cantidad, y sus
  signos opuestos se compensan. El efecto conjunto del monto es mucho menor que cualquiera de los dos.
  Lo mismo pasa con las dos columnas de `term`: solo importa su diferencia (0,91 a favor de 60 meses).
- **No identificación** de las columnas redundantes (apartado 3).

La interpretación por variable se hace con los puntos de la scorecard y la importancia por permutación
(secciones 3.7 y 5.6, P4).

## 7. Problemas de datos encontrados

Estos problemas vienen del preprocesamiento (sección 3.1, P1) y afectan a todos los modelos de PD:

- **Tres columnas constantes en entrenamiento** (apartado 3). No aportan información, inflan el conteo de
  parámetros y crean redundancia con el intercepto. Además, dos de ellas valen 1 en todo el
  entrenamiento pero no en las otras particiones: `earliest_cr_line_infrequent_sklearn` vale 0 en 377
  créditos de validación y 560 de prueba, y `application_type_INDIVIDUAL` en 3 de prueba. Para esos
  créditos, el coeficiente de −0,163, que en entrenamiento era solo una parte del intercepto, sube el
  logit en 0,163 sin que el modelo haya aprendido nada sobre ellos.
- **Valores extremos sin tratar.** En validación, `tot_coll_amt` llega a 13.329 desviaciones estándar,
  cuando su máximo en entrenamiento es 109. Ese crédito recibe una PD de 1,0. Por ese único valor, **la
  BCE de validación depende de cómo se recorten las probabilidades**: 0,54219 con el recorte de
  `log_loss` de scikit-learn (la de esta sección) y 0,54246 con el de `src/evaluation/metrics.py`
  (sección 5.3). La AUC no cambia, porque solo depende del orden.

## 8. Limitaciones

- **La selección se hizo con un solo criterio (BCE)**, y las diferencias entre las mejores
  configuraciones son menores que la tolerancia del optimizador. Cualquier $C \geq 0{,}003$ con L2 habría
  dado prácticamente el mismo modelo.
- **No se probó una logística con transformaciones de variables** (logaritmos o cortes) ni con
  interacciones. Un modelo así sería una referencia más fuerte frente a la red.
- **La calibración temporal se corrige solo en parte.** Platt (apartado 4) acerca la PD media de 2015 a
  la tasa real, pero se pasa, porque la tasa vuelve a cambiar de un año a otro. No se probó la
  recalibración isotónica.
- **Los valores extremos** (apartado 7) no se trataron.
