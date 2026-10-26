# Sección 3.6 (borrador): métricas de evaluación

**Curso:** Redes Neuronales y Algoritmos Bioinspirados
**Universidad:** Universidad Nacional de Colombia
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)
**Responsable:** P4 · **Revisión:** P3 · **Estado:** borrador; las cifras de resultados van en las secciones 5 y 6

> Implementación: `src/evaluation/metrics.py` y `src/evaluation/distributions.py`; pruebas en
> `tests/test_metrics.py` y `tests/test_distributions.py`. Todas las métricas del trabajo (búsquedas de
> hiperparámetros, 30 semillas y evaluación final) se calculan con este mismo módulo, para que las cifras
> de los distintos modelos sean comparables. La numeración de tablas es provisional (3.6.x).

**Notación.** $y_i \in \lbrace 0,1 \rbrace$ es el resultado observado (1 = incumplimiento), $\hat p_i$ la PD predicha,
$n_1$ y $n_0$ el número de malos y de buenos. En LGD, $y_i \in [0,1]$ es la LGD observada y $\hat y_i$ la
predicha.

---

## 1. Discriminación (PD)

**AUC.** Para cada umbral $t$, la curva ROC grafica la tasa de verdaderos positivos
$\text{TPR}(t) = P(\hat p > t \mid y = 1)$ contra la de falsos positivos
$\text{FPR}(t) = P(\hat p > t \mid y = 0)$; la AUC es el área bajo esa curva. Su interpretación
probabilística (Hanley y McNeil, 1982) es la que se implementa:

$$\text{AUC} = P(\hat p_{\text{malo}} > \hat p_{\text{bueno}}) + \tfrac{1}{2}\thinspace P(\hat p_{\text{malo}} = \hat p_{\text{bueno}}),$$

es decir, la probabilidad de que un mal pagador elegido al azar reciba más PD que un buen pagador elegido
al azar. Se estima con el estadístico de Mann–Whitney:

$$\widehat{\text{AUC}} = \frac{R_1 - n_1(n_1+1)/2}{n_1\thinspace n_0},$$

donde $R_1$ es la suma de los rangos de los malos en la muestra completa (rango promedio en los empates).
Como solo depende del orden, la AUC es invariante a cualquier transformación monótona creciente de la
predicción: vale lo mismo calculada sobre la PD, sobre el logit o sobre el puntaje cambiado de signo.

**Gini.** $\text{Gini} = 2\thinspace \text{AUC} - 1 \in [-1, 1]$, también llamado *accuracy ratio*. Es la forma
habitual de reportar la discriminación en riesgo de crédito. No debe confundirse con el coeficiente de
Gini de una curva de Lorenz de ingresos ni con la impureza de Gini de los árboles de decisión: comparten
el nombre, pero son objetos distintos.

**KS.** Sean $F_1(t)$ y $F_0(t)$ las funciones de distribución empíricas de $\hat p$ entre malos y entre
buenos. El estadístico de Kolmogorov–Smirnov es

$$\text{KS} = \max_t \thinspace \lvert F_1(t) - F_0(t) \rvert,$$

la máxima separación entre ambas distribuciones. Cuando el modelo asigna más PD a los malos,
$F_0(t) - F_1(t) = \text{TPR}(t) - \text{FPR}(t)$, de modo que el KS es el máximo del estadístico J de
Youden sobre todos los umbrales. Se evalúa en los valores observados de $\hat p$ y coincide con el
estadístico de dos muestras de Kolmogorov–Smirnov entre las predicciones de malos y de buenos.

## 2. Calibración (PD)

**Puntaje de Brier.** $\text{BS} = \frac{1}{n}\sum_i (\hat p_i - y_i)^2$ (Brier, 1950). Penaliza tanto
discriminar mal como estar mal calibrado. Si las predicciones se agrupan en $K$ grupos con $n_k$ créditos,
PD media $\bar p_k$ y tasa observada $\bar o_k$, y $\bar o$ es la tasa global, la descomposición de Murphy
(1973) separa ambos componentes:

$$\text{BS} \approx \underbrace{\tfrac{1}{n}\textstyle\sum_k n_k(\bar p_k - \bar o_k)^2}_{\text{fiabilidad}}
\enspace -\enspace  \underbrace{\tfrac{1}{n}\textstyle\sum_k n_k(\bar o_k - \bar o)^2}_{\text{resolución}}
\enspace +\enspace  \underbrace{\bar o(1-\bar o)}_{\text{incertidumbre}}.$$

La igualdad es exacta cuando todas las predicciones de un grupo son iguales. Un modelo mejor calibrado
reduce la fiabilidad; uno que discrimina mejor aumenta la resolución.

**Entropía cruzada binaria.** $\text{BCE} = -\frac{1}{n}\sum_i \big[y_i \ln \hat p_i + (1-y_i)\ln(1-\hat p_i)\big]$
es la log-verosimilitud negativa media de un modelo Bernoulli; su derivación por máxima verosimilitud está
en la sección 3.3. En la implementación, $\hat p$ se recorta a $[\varepsilon, 1-\varepsilon]$, con
$\varepsilon$ el épsilon de máquina, para evitar $\ln 0$.

**Curva de calibración.** Los créditos se agrupan en bins por cuantiles de $\hat p$ (igual número de
créditos por bin) y en cada uno se compara la PD media $\bar p_k$ con la tasa observada $\bar o_k$. Un
modelo bien calibrado sigue la diagonal. Se usan cuantiles y no bins de igual ancho porque, con una tasa
de incumplimiento del 19 % al 25 %, los bins de PD alta quedarían casi vacíos.

## 3. Regresión (LGD)

$$\text{MAE} = \frac{1}{n}\sum_i \lvert y_i - \hat y_i \rvert, \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum_i (y_i - \hat y_i)^2}, \qquad
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.$$

**Calibración de un predictor continuo.** Es la pendiente $b$ de la regresión de $y$ sobre $\hat y$:
$b = \mathrm{Cov}(y, \hat y)/\mathrm{Var}(\hat y)$. Con calibración perfecta $b = 1$ y el
intercepto es 0; $b < 1$ indica predicciones demasiado dispersas y $b > 1$, demasiado comprimidas. Se
complementa con la calibración por deciles de $\hat y$ (media predicha frente a media observada).

**Comparación de distribuciones.** Se reportan: el estadístico de Kolmogorov–Smirnov de dos muestras
$D = \sup_t \lvert F_y(t) - F_{\hat y}(t)\rvert$; la distancia de Wasserstein
$W_1 = \int_0^1 \lvert Q_y(u) - Q_{\hat y}(u)\rvert\thinspace du$, donde $Q$ es la función cuantil; el cociente
$\mathrm{sd}(\hat y)/\mathrm{sd}(y)$; y la fracción de créditos cuya LGD observada cae fuera
del rango $[\min \hat y, \max \hat y]$ que el modelo alcanza a predecir.

**Proposición (compresión esperada).** Si $\hat y = E[Y \mid X]$, entonces
(i) $E[\hat y] = E[Y]$; (ii) $\mathrm{Cov}(Y, \hat y) = \mathrm{Var}(\hat y)$, de modo que $b = 1$;
y (iii) $R^2 = \mathrm{Var}(\hat y)/\mathrm{Var}(Y)$, es decir,
$\mathrm{sd}(\hat y)/\mathrm{sd}(Y) = \sqrt{R^2}$.

*Demostración.* (i) es la ley de la esperanza total. Para (ii), con $\mu = E[Y]$,
$\mathrm{Cov}(Y, \hat y) = E\big[E[Y - \mu \mid X]\thinspace (\hat y - \mu)\big] = E[(\hat y - \mu)^2] = \mathrm{Var}(\hat y)$.
Para (iii), el error cuadrático medio es $E[(Y - \hat y)^2] = E[\mathrm{Var}(Y \mid X)]$ y, por la ley de
la varianza total, $\mathrm{Var}(Y) = E[\mathrm{Var}(Y \mid X)] + \mathrm{Var}(\hat y)$; luego
$R^2 = 1 - E[\mathrm{Var}(Y\mid X)]/\mathrm{Var}(Y) = \mathrm{Var}(\hat y)/\mathrm{Var}(Y)$. $\blacksquare$

*Consecuencia.* Un modelo que predice la media condicional **no** debe reproducir la distribución observada:
con $R^2 \approx 0{,}21$ se espera $\mathrm{sd}(\hat y) \approx 0{,}46\thinspace \mathrm{sd}(y)$. Por eso la
comparación de distribuciones se interpreta frente a $\sqrt{R^2}$ y la calibración se juzga con $b$ y con los
deciles. Usar el parecido entre distribuciones como criterio de selección premiaría modelos
sobredispersos, es decir, peor calibrados. La propiedad se verifica numéricamente en
`test_predictor_calibrado_cociente_sd_es_raiz_r2`.

## 4. Variabilidad entre semillas

Para cada red se entrena la configuración final con 30 semillas y, para cada métrica, se reportan la
media, la desviación estándar muestral ($n-1$), el mejor y el peor valor. Los modelos de referencia
(regresión logística y lineal) son deterministas y se reportan con una sola corrida.

## 5. Verificación de la implementación

Las funciones propias se contrastan con `scikit-learn` y `scipy` (diferencias menores que $10^{-12}$,
incluidos datos con muchos empates) y con valores teóricos. Si $\hat p \sim U(0,1)$ y
$y \mid \hat p \sim \text{Bernoulli}(\hat p)$, las densidades de $\hat p$ entre malos y entre buenos son
$2p$ y $2(1-p)$, y los valores exactos de la Tabla 3.6.1 se obtienen integrando.

| Métrica | Valor teórico | Derivación |
|---|---|---|
| AUC | $5/6 \approx 0{,}833$ | $\int_0^1\int_0^a 2a \cdot 2(1-b)\thinspace db\thinspace da$ |
| KS | $1/2$ | $F_1(t) = t^2$, $F_0(t) = 2t - t^2$; la diferencia $2t(1-t)$ es máxima en $t = 1/2$ |
| Brier | $1/6 \approx 0{,}167$ | $E[\hat p(1-\hat p)]$ |
| BCE | $1/2$ | $2\int_0^1 -p\ln p\thinspace dp$ |

*Tabla 3.6.1. Valores teóricos de las métricas para un modelo perfectamente calibrado con
$\hat p \sim U(0,1)$. Fuente: elaboración propia; `test_valores_teoricos_uniforme_bernoulli` los reproduce
con $n = 400\thinspace 000$ y tolerancia de $\pm 0{,}005$.*

---

### Referencias candidatas (verificar autores, año, volumen, páginas y DOI en la fuente primaria)

- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review, 78*(1), 1–3.
- Hanley, J. A., y McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology, 143*(1), 29–36.
- Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology, 12*(4), 595–600.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
