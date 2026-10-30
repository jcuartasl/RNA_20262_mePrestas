# 3.3 Funciones de pérdida por máxima verosimilitud

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  
**Responsable:** P3 · **Revisión:** P2 · **Estado:** borrador para integración

> La sección tiene dos partes. Los apartados 1 a 5 derivan cada función de pérdida que usa el proyecto a
> partir de un modelo probabilístico. Los apartados 6 a 11 aplican esas derivaciones a la LGD: fijan la
> activación de salida, las candidatas, el criterio de selección y unas predicciones, **antes** de la
> búsqueda de hiperparámetros, para que la elección no se justifique a posteriori con el resultado que
> salió mejor por azar. Los resultados se reportan en la sección 6.
>
> *Nota de versión:* las cifras de los apartados 6 a 11 se recalcularon cuando el equipo volvió a sortear
> la partición aleatoria (`split_seed: 90`, sección 4). Ninguna conclusión cambió. La numeración de
> figuras y tablas (3.3.x) es provisional hasta la integración del blog.

---

## 1. Principio: de un modelo probabilístico a una pérdida

Sean $n$ observaciones independientes $(x_i, y_i)$ y un modelo $p(y \mid x; \theta)$. La máxima
verosimilitud elige $\theta$ maximizando $\prod_i p(y_i \mid x_i; \theta)$, lo que equivale a minimizar
la log-verosimilitud negativa promedio (NLL):

$$\ell(\theta) = -\frac{1}{n}\sum_{i=1}^{n} \ln p(y_i \mid x_i; \theta).$$

En todos los modelos del proyecto, la red (o la regresión) produce un número real $z_i = f(x_i; \theta)$,
el *logit*, y el parámetro de la distribución se obtiene de él. Por la regla de la cadena,

$$\frac{\partial \ell}{\partial \theta} = \frac{1}{n}\sum_{i=1}^{n} \frac{\partial \ell_i}{\partial z_i}\,\frac{\partial z_i}{\partial \theta},$$

donde $\ell_i$ es el término del dato $i$. El segundo factor lo calcula la retropropagación (sección 3.4);
lo que cambia de una pérdida a otra es el primero, $\partial \ell_i / \partial z_i$, que se deriva abajo
para cada caso. Se usará la sigmoide $\sigma(z) = 1/(1+e^{-z})$ y su derivada
$\sigma'(z) = \sigma(z)\,(1-\sigma(z))$.

## 2. Bernoulli → entropía cruzada binaria (BCE): PD

**Modelo.** $Y \mid x \sim \text{Bernoulli}(\mu)$ con $\mu = \sigma(z)$, de modo que
$p(y \mid x) = \mu^{y}(1-\mu)^{1-y}$ para $y \in \lbrace 0, 1 \rbrace$.

**Pérdida.** Tomando $-\ln p$:

$$\ell_i = -\left[\, y_i \ln \mu_i + (1-y_i)\ln(1-\mu_i) \,\right].$$

**Forma estable.** Como $\ln \sigma(z) = -\ln(1+e^{-z})$, $\ln(1-\sigma(z)) = -\ln(1+e^{z})$ y
$\ln(1+e^{-z}) = \ln(1+e^{z}) - z$,

$$\ell_i = \ln\left(1+e^{z_i}\right) - y_i z_i .$$

Esta es la forma que se implementa (`bce_con_logits` en `src/models/losses.py`), con
$\ln(1+e^{z})$ calculado como `logaddexp(0, z)`: no hay desbordamiento con logits grandes y se evita
calcular $\ln 0$ cuando la sigmoide se redondea a 0 o a 1.

**Gradiente.** $\dfrac{\partial \ell_i}{\partial z_i} = \sigma(z_i) - y_i = \mu_i - y_i$. No contiene el
factor $\sigma'(z)$, así que no se apaga cuando la red está muy segura y se equivoca.

**Regresión logística.** Es el caso $z = \beta_0 + \beta^\top x$: la logística de referencia de PD se
estima minimizando exactamente esta pérdida (más la penalización de la sección 3.5). Su hessiano,
$\frac{1}{n}\sum_i \mu_i(1-\mu_i)\,x_i x_i^\top$, es semidefinido positivo, así que el problema es convexo.
La red de PD usa la misma pérdida con $z$ dado por el MLP.

## 3. Normal → error cuadrático medio (MSE)

**Modelo.** $Y \mid x \sim \mathcal{N}(\mu, s^2)$ con varianza $s^2$ constante:

$$\ell_i = \tfrac{1}{2}\ln(2\pi s^2) + \frac{(y_i - \mu_i)^2}{2 s^2}.$$

Con $s^2$ fijo, el primer término no depende de $\theta$ y el factor $1/(2s^2)$ no cambia el
minimizador, así que minimizar la NLL equivale a minimizar $\frac{1}{n}\sum_i (y_i - \mu_i)^2$.

- **Regresión lineal** (referencia de LGD, sección 6): $\mu = \beta_0 + \beta^\top x$, es decir, mínimos
  cuadrados ordinarios.
- **Red de LGD con salida sigmoide** (`mse_con_logits`): $\mu = \sigma(z)$ y

$$\frac{\partial \ell_i}{\partial z_i} = 2\,(\mu_i - y_i)\,\mu_i(1-\mu_i).$$

El factor $\mu(1-\mu)$ tiende a 0 cuando la sigmoide satura: si la red predice 0,999 para un crédito con
LGD 0,1, el gradiente es casi nulo (`test_mse_se_satura_con_logits_extremos`).

## 4. BCE fraccional: cuasi-verosimilitud para $y \in [0, 1]$

**Pérdida.** La misma expresión de la BCE, $\ell_i = \ln(1+e^{z_i}) - y_i z_i$, pero con $y_i$ continua
en $[0, 1]$ (`bce_fraccional_con_logits`, que es la que devuelve `get_loss("bce")`).

**No es una verosimilitud.** $\mu^{y}(1-\mu)^{1-y}$ no integra 1 sobre $[0, 1]$:
$\int_0^1 \mu^{y}(1-\mu)^{1-y}\,dy = (2\mu - 1)/\ln\frac{\mu}{1-\mu}$, que vale 0,5 en $\mu = 0{,}5$. Por
eso se habla de cuasi-verosimilitud: solo se especifica la media, $E[Y \mid x] = \sigma(z)$, y la forma
de la pérdida corresponde a suponer $\text{Var}(Y \mid x) \propto \mu(1-\mu)$.

**Por qué funciona.** Fijado $x$, sea $m = E[Y \mid x]$. La pérdida esperada es
$-[\,m \ln \mu + (1-m)\ln(1-\mu)\,]$ y su derivada respecto a $\mu$,
$-m/\mu + (1-m)/(1-\mu)$, se anula en $\mu = m$. El minimizador es la media condicional **cualquiera
que sea la distribución de $Y$** (`test_bce_fraccional_estima_la_media`).

**Gradiente.** Igual que en la BCE: $\partial \ell_i / \partial z_i = \mu_i - y_i$. Acepta 0 y 1
exactos sin tratamiento especial.

## 5. Beta: regresión Beta con media y precisión

**Modelo.** $Y \mid x \sim \text{Beta}(a, b)$ con $a = \mu\phi$, $b = (1-\mu)\phi$, $\mu = \sigma(z)$ y
una precisión $\phi > 0$ común que se aprende como $\phi = e^{\eta}$. Así $E[Y] = \mu$ y
$\text{Var}(Y) = \mu(1-\mu)/(1+\phi)$. La densidad es

$$p(y) = \frac{\Gamma(\phi)}{\Gamma(a)\,\Gamma(b)}\, y^{a-1}(1-y)^{b-1}, \qquad 0 < y < 1,$$

y la pérdida (`beta_con_logits`)

$$\ell_i = \ln\Gamma(a_i) + \ln\Gamma(b_i) - \ln\Gamma(\phi) - (a_i - 1)\ln y_i - (b_i - 1)\ln(1-y_i).$$

**Gradientes.** Con $\psi$ la función digamma y usando $\partial a/\partial z = \phi\mu(1-\mu)$ y
$\partial b/\partial z = -\phi\mu(1-\mu)$:

$$\frac{\partial \ell_i}{\partial z_i} = \phi\,\mu_i(1-\mu_i)\left[\left(\psi(a_i) - \ln y_i\right) - \left(\psi(b_i) - \ln(1-y_i)\right)\right],$$

$$\frac{\partial \ell_i}{\partial \phi} = \mu_i\psi(a_i) + (1-\mu_i)\psi(b_i) - \psi(\phi) - \mu_i\ln y_i - (1-\mu_i)\ln(1-y_i),
\qquad \frac{\partial \ell_i}{\partial \eta} = \phi\,\frac{\partial \ell_i}{\partial \phi}.$$

Como $E[\ln Y] = \psi(a) - \psi(\phi)$ y $E[\ln(1-Y)] = \psi(b) - \psi(\phi)$, el primer gradiente se
anula en promedio cuando el modelo reproduce $E[\ln Y - \ln(1-Y)]$: la máxima verosimilitud de la Beta
iguala **log-momentos**, no la media (consecuencia que se verifica con datos en el apartado 8).

**Ceros y unos.** $\ln 0$ no existe, así que $y$ se recorta a $[\varepsilon, 1-\varepsilon]$ con
$\varepsilon = 10^{-4}$; la sensibilidad a $\varepsilon$ se reporta en la sección 6.

**Verificación.** Las cuatro pérdidas y sus gradientes analíticos se compararon con la diferenciación
automática de PyTorch y con diferencias finitas (`gradcheck`), y los valores con implementaciones de
referencia (PyTorch para BCE y MSE, SciPy para la Beta) en `tests/test_losses.py`.

---

## 6. La LGD: variable objetivo y su distribución

La LGD se construyó con la opción A (sección 3.1). Para los créditos incumplidos,
LGD $= \text{clip}\big((F - P - R + C)/F,\ 0,\ 1\big)$, con $F$ = `funded_amnt`, $P$ = `total_rec_prncp`,
$R$ = `recoveries` y $C$ = `collection_recovery_fee`. El análisis de los apartados 6 a 9 usa la
partición aleatoria de entrenamiento ($n = 41.173$).

**Forma.** Media 0,704, desviación estándar 0,205 y **asimetría −0,99**: una cola larga hacia 0 y la
masa concentrada entre 0,6 y 0,95. Solo el 0,69 % de los valores son exactamente 0 o 1 (31 y 253
créditos). Como se observa en la Figura 3.3.1, la distribución es unimodal, con el pico entre 0,78 y
0,80, sin las masas en los extremos que a veces se describen para la LGD.

![Figura 3.3.1](../../results/figures/lgd_ajuste_beta_normal.png)

**Figura 3.3.1.** LGD observada frente a las densidades Beta y normal ajustadas por máxima
verosimilitud. Las bandas rojas marcan valores imposibles (fuera de [0, 1]).
*Fuente:* elaboración propia con datos de Lending Club (Kaggle), partición aleatoria de
entrenamiento de LGD; código en `src/models/lgd_mlp.py` (`figuras_distribucion_lgd`).

**Ajuste de dos distribuciones candidatas.** Se ajustaron por máxima verosimilitud una normal y una
Beta con media y precisión constantes. La Beta se ajustó con la misma pérdida que usa la red
(`beta_con_logits`) y dio $\mu = 0{,}694$ y $\phi = 3{,}82$. La Tabla 3.3.1 compara los cuantiles.

| Cuantil | LGD observada | Beta ajustada | Normal ajustada |
|---|---|---|---|
| 1 % | 0,101 | 0,160 | 0,227 |
| 5 % | 0,286 | 0,296 | 0,367 |
| 25 % | 0,594 | 0,553 | 0,566 |
| 50 % | 0,750 | 0,731 | 0,704 |
| 75 % | 0,858 | 0,867 | 0,842 |
| 95 % | 0,956 | 0,969 | 1,041 |
| 99 % | 0,988 | 0,992 | 1,180 |

**Tabla 3.3.1.** Cuantiles de la LGD observada y de las distribuciones ajustadas. *Fuente:* elaboración
propia, misma partición y código que la Figura 3.3.1.

- La normal asigna un **7,4 % de su masa fuera de [0, 1]**; su percentil 99 es 1,18, una LGD imposible.
- La log-verosimilitud media es 0,283 con la Beta y 0,167 con la normal.
- La Beta no es perfecta: sitúa la moda en 0,91 cuando el pico observado está entre 0,78 y 0,80, y
  sobrestima la cola baja (percentil 1: 0,160 frente a 0,101).

La Figura 3.3.2 lo muestra en forma de gráfico QQ: frente a la Beta, los cuantiles observados siguen la
diagonal salvo en la cola baja; frente a la normal describen una S y se salen del intervalo [0, 1].

![Figura 3.3.2](../../results/figures/lgd_qq_beta_normal.png)

**Figura 3.3.2.** Gráficos QQ de la LGD observada contra la Beta (izquierda) y la normal (derecha)
ajustadas. *Fuente:* elaboración propia, misma partición y código que la Figura 3.3.1.

**La varianza depende de la media.** Para ver cómo cambia la dispersión con el nivel de la LGD, se
agruparon los créditos en deciles de la predicción de la regresión lineal de referencia y se calculó
la varianza de la LGD dentro de cada decil. Como se observa en la Figura 3.3.3, la varianza **cae de
0,051 a 0,010** (unas cinco veces) cuando la LGD media del decil pasa de 0,58 a 0,86.

![Figura 3.3.3](../../results/figures/lgd_varianza_por_decil.png)

**Figura 3.3.3.** Varianza de la LGD dentro de cada decil de la predicción lineal, frente a la forma
que supone cada función de pérdida (cada supuesto se escala a la varianza promedio observada, de modo
que solo se compara la forma). *Fuente:* elaboración propia, misma partición y código que la
Figura 3.3.1.

**Las variables explican una parte pequeña de la LGD.** La regresión lineal de referencia (P2) obtiene
$R^2 = 0{,}19$ en validación, y sus predicciones tienen una desviación estándar de 0,088 frente a 0,202
de la LGD observada en ese mismo conjunto.

## 7. Activación de salida: sigmoide

La LGD pertenece a [0, 1], así que la salida de la red debe estar en ese intervalo. Se usa la sigmoide,
$\mu = \sigma(z)$, aplicada al logit $z$ de la última capa lineal. La red entrega $z$ y la sigmoide se
aplica dentro de la función de pérdida y al predecir: es la misma salida, pero permite calcular de forma
estable las pérdidas que contienen logaritmos (apartado 2). Las tres candidatas comparten esta salida,
de modo que la única diferencia entre ellas es el modelo probabilístico del error.

## 8. Las tres candidatas frente a los datos

- **MSE (apartado 3).** A favor: su minimizador es $E[Y \mid x]$ sea cual sea la distribución real, y
  optimiza directamente el RMSE. En contra: supone varianza constante, y la Figura 3.3.3 muestra que la
  varianza cae unas cinco veces; el modelo normal pone probabilidad fuera de [0, 1] (Tabla 3.3.1); y con
  salida sigmoide el gradiente se apaga por el factor $\mu(1-\mu)$.
- **BCE fraccional (apartado 4).** A favor: también estima $E[Y \mid x]$ para cualquier distribución;
  acepta 0 y 1; su gradiente no se satura; y su varianza implícita $\mu(1-\mu)$ va en la dirección de la
  observada. En contra: esa forma capta solo parte de la caída (Figura 3.3.3); no da una distribución
  predictiva completa; y su valor tiene un piso igual a la entropía de $y$, así que no es comparable con
  el de las otras pérdidas.
- **Beta (apartado 5).** A favor: es una verosimilitud verdadera en (0, 1), describe la distribución
  marginal mucho mejor que la normal (Tabla 3.3.1, Figuras 3.3.1 y 3.3.2), da una distribución predictiva
  completa y su gradiente no se satura. En contra:
  - No admite 0 ni 1 exactos. Con el recorte, esos puntos (0,69 % de los datos) entran con
    $\ln\varepsilon$, mucho más que un dato típico ($\ln 0{,}75 = -0{,}29$). En el ajuste sin variables
    mueven sobre todo $\phi$ (de 4,12 con $\varepsilon = 10^{-2}$ a 3,71 con
    $\varepsilon = 1{,}2\times10^{-5}$) y casi nada la media ajustada (entre 0,6943 y 0,6946).
  - Con $\phi$ constante supone la misma forma de varianza que la BCE fraccional, que se queda corta
    frente a la caída observada (Figura 3.3.3).
  - **No iguala la media muestral.** Como se vio en el apartado 5, la máxima verosimilitud iguala los
    log-momentos. Solo si los datos siguen una Beta eso equivale a estimar bien la media. Con datos
    simulados de una Beta, la media ajustada coincide con la muestral (diferencia +0,0002); con la LGD
    observada, sin variables, queda 0,0096 por debajo (0,6943 frente a 0,7039), y 0,0099 por debajo si se
    excluyen los 0 y 1 exactos, así que la diferencia se debe a la forma de la distribución y no al
    recorte.

## 9. Criterio de selección (fijado antes de entrenar)

1. **Métrica principal:** RMSE en la **validación aleatoria**. Las tres pérdidas se comparan por la
   métrica de evaluación y no por sus propios valores, que están en escalas distintas.
2. **Partición:** aleatoria. En la temporal, la LGD media es 0,615 en entrenamiento, 0,768 en validación
   y 0,905 en prueba, por la censura de las recuperaciones de los créditos recientes (sección 3.1);
   seleccionar con ella castigaría a cualquier modelo por un artefacto de los datos. La partición
   temporal se reporta como análisis de estabilidad.
3. **Desempate:** comparación de la distribución de las predicciones con la distribución observada,
   como pide el enunciado (histograma y cuantiles de las predicciones puntuales de cada modelo).
4. **Presupuesto:** las tres pérdidas entran como un hiperparámetro más en la misma búsqueda, con el
   mismo presupuesto, que se declara en la sección 4.
5. **Conjunto de prueba:** no se usa para seleccionar; se evalúa una sola vez al final.

*Nota añadida después de la búsqueda.* El punto 3 no decía qué diferencia cuenta como empate. Al
aplicarlo se fijó una tolerancia de $2{,}5\times10^{-4}$ de RMSE, la desviación entre semillas medida en
la primera corrida, y la comparación de distribuciones se resumió en la búsqueda con la desviación
estándar de las predicciones (todas las redes encogen, así que la de mayor dispersión es la más cercana
a la observada). En la sección 6 se comparan además el estadístico KS y la distancia de Wasserstein.

## 10. Predicciones escritas antes de entrenar

Se contrastan con los resultados en la sección 6.

1. **RMSE parecido entre las tres pérdidas.** Las tres estiman (o aproximan) la media condicional, y el
   $R^2$ de la lineal (0,19) sugiere que el techo lo pone la información de las variables más que la
   pérdida.
2. **Las predicciones de las tres estarán encogidas** respecto de la LGD observada (desviación estándar
   muy por debajo de 0,20, como ya ocurre con la lineal). Es una propiedad de predecir medias
   condicionales, no un defecto de una pérdida en particular.
3. **La MSE no debería superar a la BCE fraccional**, porque supone varianza constante cuando la
   varianza real cambia con la media; la diferencia, si la hay, debería verse en los créditos de LGD alta.
4. **La Beta será la que mejor describa la forma de la distribución**, pero no necesariamente la de
   menor RMSE.

## 11. Limitaciones y ajustes posibles

- **Precisión no constante.** La varianza observada cae más rápido que $\mu(1-\mu)$ (Figura 3.3.3), lo
  que sugiere que $\phi$ podría depender de las variables. Modelarlo exigiría una segunda salida en la
  red; queda como posible ajuste si la distribución predicha no reproduce la observada.
- **Sensibilidad a $\varepsilon$** en la Beta: se evalúa con
  $\varepsilon \in \lbrace 10^{-2}, 10^{-3}, 10^{-4}, 1{,}2\times10^{-5}\rbrace$ (el último equivale a la
  transformación de Smithson y Verkuilen con el $n$ de entrenamiento).
- **Análisis marginal.** La Tabla 3.3.1 y las Figuras 3.3.1 y 3.3.2 describen la distribución de la LGD
  sin condicionar a las variables; la red ajusta una distribución por crédito, así que la evidencia
  decisiva es la comparación de la sección 6.
- **Censura temporal** de las recuperaciones (sección 3.1).

---

## Referencias por verificar

Antes de citarlas en la versión final deben confirmarse autores, año, título y DOI en la fuente
primaria:

- Ferrari y Cribari-Neto (2004), sobre regresión Beta con parametrización media–precisión.
- Papke y Wooldridge (1996), sobre modelos para respuestas fraccionales.
- Smithson y Verkuilen (2006), sobre la transformación de valores 0 y 1 en regresión Beta.
- Wedderburn (1974), sobre cuasi-verosimilitud.
