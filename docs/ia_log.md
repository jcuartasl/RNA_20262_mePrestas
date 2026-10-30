# Registro de uso de IA y cacería de alucinaciones

## P3 — Juan David Robayo (pérdidas, optimización y regularización)

Cada hallazgo sigue los seis puntos del enunciado. Las cifras se recalcularon con la partición aleatoria
vigente (`split_seed: 90`). Cuando la respuesta de la IA cita números, son los de la partición que había
en ese momento.

---

### P3-1. "La L2 dentro de Adam no equivale a sumar λ‖w‖² a la pérdida"

- **Categoría:** matemáticas / conceptos (optimizador y regularización)
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-24
- **1. Prompt (literal):**
  > bien, parece que mis compañeros ya hicieron varios merge a develop, quieres verlo para que sepas
  > en que avance va el grupo?
- **2. Respuesta de la IA (parte relevante, literal):**
  > 7. **L2 implementada como `weight_decay` dentro de Adam.** Con Adam eso no equivale exactamente a
  > sumar λ‖w‖² a la pérdida; esa es la diferencia entre Adam y AdamW. Es un buen punto para tu
  > sección 3.5 y posible material para la cacería de alucinaciones.
- **3. Sospecha:** el gradiente de $\tfrac{\lambda}{2}\lVert w\rVert^2$ es $\lambda w$. Si `weight_decay`
  suma $\lambda w$ al gradiente antes de los momentos, las dos cosas deberían ser idénticas.
- **4. Evidencia:**
  - En el código fuente de PyTorch 2.14.1 (`torch/optim/adam.py`), con `weight_decay` y sin
    `decoupled_weight_decay`, Adam hace `grad = grad.add(param, alpha=weight_decay)` antes de actualizar
    los momentos.
  - Experimento `tests/test_optim.py::test_weight_decay_de_adam_es_l2_y_adamw_no`: MLP 5→8→1 en float64,
    $\lambda = 0{,}1$, 30 pasos desde los mismos pesos. `Adam(weight_decay=0,1)` y Adam sobre
    `bce + 0,1 · penalizacion_l2` difieren como máximo en $5{,}6\times10^{-17}$ (iguales a precisión de
    máquina en cada paso). `AdamW` se separa en 0,58.
  - Comando: `python -m pytest tests/test_optim.py -k weight_decay`.
- **5. Corrección:** dentro de Adam, el `weight_decay` acoplado **sí** equivale a una L2 en la pérdida
  (con el factor ½). Lo que no es equivalente es la L2 frente al decaimiento desacoplado (AdamW). El
  único reparo real al uso de `weight_decay` en la red de PD es que también penaliza los sesgos
  (sección 3.5).
- **6. Lección:** cuando una IA dice "A no equivale a B", escribir las dos actualizaciones y
  comprobarlas con un experimento de pocos pasos antes de repetirlo en el reporte.

---

### P3-2. "La MSE tiende a encoger las predicciones hacia la media" (como defecto propio de la MSE)

- **Categoría:** conceptos
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-24
- **1. Prompt (literal; mensaje largo, se cita su comienzo):**
  > Antes de escribir código, quiero confirmar que entendiste mi rol y acordar un plan...
- **2. Respuesta de la IA (parte relevante, literal):**
  > **MSE**: simple y coincide con la métrica RMSE, pero supone ruido gaussiano y tiende a encoger las
  > predicciones hacia la media.
- **3. Sospecha:** cualquier modelo que predice la media condicional $E[Y\mid x]$ produce predicciones
  menos dispersas que los datos, porque no puede predecir la parte de $Y$ que las variables no explican.
  No es una propiedad de la MSE en particular.
- **4. Evidencia:** 30 semillas por pérdida en la validación aleatoria de LGD. La desviación estándar de
  la LGD observada es 0,202. La de las predicciones es 0,097 con BCE fraccional, 0,096 con MSE, 0,087 con
  Beta y 0,088 con la regresión lineal. Las tres pérdidas encogen de forma parecida, y la Beta incluso más
  que la MSE (`results/tables/lgd_semillas_resumen.csv`; sección 6.3, Tabla 6.3.3).
- **5. Corrección:** encoger la distribución es una propiedad de predecir medias condicionales. Las
  pérdidas difieren en cómo ponderan los errores, y solo la Beta da una distribución completa.
- **6. Lección:** separar "propiedad de la pérdida" de "propiedad de predecir una media"; comprobarlo
  comparando todas las pérdidas sobre los mismos datos.

---

### P3-3. "La Beta con φ constante es exacta en los deciles bajos" (comparación mal planteada)

- **Categoría:** conceptos (varianza condicional frente a marginal)
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-25
- **1. Prompt (literal):**
  > sigamos con la etapa 4
- **2. Respuesta de la IA (parte relevante, literal):** una tabla con la columna "Varianza Beta:
  μ(1−μ)/(1+φ)", calculada con el φ = 3,83 del ajuste marginal, y la conclusión:
  > La Beta con φ constante acierta la dirección y es exacta en los deciles bajos, pero **se queda
  > corta en los altos**: la varianza real cae más rápido.
- **3. Sospecha:** el φ viene del ajuste **marginal** de la Beta, sin variables, que mezcla la varianza
  que explican las variables con la que no explican. La varianza **dentro** de cada decil es condicional
  y no tiene por qué coincidir con $\mu(1-\mu)/(1+\phi_{\text{marginal}})$.
- **4. Evidencia:** con los datos actuales, φ marginal = 3,815. En el decil 1 la Beta da 0,0506 y la
  varianza observada es 0,0508; en el decil 5, 0,0447 frente a 0,0396. La coincidencia en el decil 1 es
  casual. Para comparar solo la forma, el cociente varianza observada / $\mu(1-\mu)$ cae de 0,208 en el
  decil 1 a 0,085 en el decil 10: la varianza real cae unas 2,5 veces más rápido que la forma
  $\mu(1-\mu)$. (Deciles de la predicción lineal en el entrenamiento aleatorio; figura
  `results/figures/lgd_varianza_por_decil.png`, sección 3.3.)
- **5. Corrección:** la BCE fraccional y la Beta con φ constante suponen la misma forma de varianza,
  proporcional a $\mu(1-\mu)$. Esa forma capta parte de la caída observada, pero la varianza real cae más
  rápido. No hay base para decir que la Beta es "exacta" en ningún decil.
- **6. Lección:** no comparar un parámetro estimado sin variables (marginal) con una cantidad
  condicional; cuando solo interesa la forma, normalizar antes de comparar.

---

### P3-4. "El techo de la tanh subestimaría de forma sistemática los créditos de LGD alta"

- **Categoría:** conceptos (confundir error individual con sesgo de la media condicional)
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-25
- **1. Prompt (literal):**
  > si, pero antes, cuando se concluya esos 75 min que faltaria? hago pr ya?
- **2. Respuesta de la IA (parte relevante, literal; la IA la marcó como hipótesis sin verificar):**
  > Mi hipótesis, **todavía sin verificar**: las dos usan la configuración 9, con activación **tanh**,
  > que está acotada entre −1 y 1. Si las neuronas de la última capa oculta se saturan, el logit de
  > salida tiene un máximo y la predicción no puede subir más. Esto subestimaría de forma sistemática
  > los créditos de LGD alta, que son justamente los más costosos para la pérdida esperada.
- **3. Sospecha:** un modelo que predice medias condicionales no debería predecir 0,95 si las variables
  no distinguen qué créditos terminan con una LGD tan alta. Que las predicciones individuales no lleguen a
  esos valores no implica un sesgo.
- **4. Evidencia** (`src/search/lgd_random_search.py --fase diagnostico`; sección 6.3, apartado 7):
  - **El mecanismo existe.** El techo teórico $\sigma(b + \sum_j \lvert w_j\rvert)$ vale 0,877 en promedio
    en los 30 modelos y la predicción máxima es 0,875. En el 1 % de predicciones más altas, el 44 % de las
    neuronas tanh están saturadas (19 % en el resto).
  - **La consecuencia no.** En el decil más alto de la predicción, la BCE predice 0,864 y lo observado es
    0,868 (bien calibrada). Los créditos con LGD observada mayor que 0,9 (16,5 % de la validación) se
    predicen en promedio entre 0,744 y 0,763 con **todos** los modelos, incluida la lineal (0,757), que
    no tiene techo.
  - **Experimento:** la misma configuración con ReLU (sin techo), 5 semillas, da un RMSE medio de 0,17888
    frente a 0,17734 con tanh; la ReLU es peor en las 5 semillas. Quitar el techo no mejora nada, aunque
    la configuración se eligió con tanh.
  - Tablas `results/tables/lgd_diagnostico_techo.csv`, `lgd_calibracion_deciles.csv` y
    `lgd_techo_relu.csv`.
- **5. Corrección:** el mecanismo propuesto (saturación de la tanh) es real, pero no sesga la media
  condicional. La subpredicción de los créditos de LGD muy alta es el encogimiento hacia la media
  condicional (P3-2), común a todos los modelos.
- **6. Lección:** para afirmar que un modelo "subestima" un grupo hay que comparar medias predichas y
  observadas dentro de grupos definidos por la **predicción** (calibración por deciles), no mirar los
  valores individuales extremos.

---

### P3-5. "La Beta subestima por la asimetría a la izquierda; esto explica su peor RMSE" (causa sin evidencia)

- **Categoría:** conceptos (atribuir una causa sin comprobarla)
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-25
- **1. Prompt (literal):**
  > sigamos con la etapa 7
- **2. Respuesta de la IA (parte relevante, literal):**
  > Encaja con lo que dijimos en la Etapa 1: si los datos no son Beta, su ecuación iguala E[logit Y] y
  > no E[Y], y con una LGD asimétrica a la izquierda eso tira la predicción hacia abajo. **Esto explica
  > por qué la Beta tiene peor RMSE.**
- **3. Sospecha:** se afirmó una causa (la asimetría) y una explicación completa del peor RMSE sin
  haberlas probado.
- **4. Evidencia** (ajuste marginal de la Beta con `ajustar_beta_marginal` de `src/models/lgd_mlp.py`,
  entrenamiento aleatorio, 41.173 créditos):
  - **La parte matemática es correcta, pero en otra forma.** La MLE de la Beta no iguala la media de $y$
    sino sus log-momentos. Con $\varepsilon = 10^{-4}$: $\psi(\alpha) - \psi(\alpha+\beta) = -0{,}42850$,
    igual a la media de $\ln y$, y $\psi(\beta) - \psi(\alpha+\beta) = -1{,}53511$, igual a la media de
    $\ln(1-y)$.
  - **Control.** Con datos simulados de una Beta, la media ajustada coincide con la muestral (diferencia
    +0,0001). Con la LGD real queda 0,0095 por debajo de la media (0,6944 frente a 0,7039), con
    $\varepsilon$ = $10^{-2}$, $10^{-3}$ o $10^{-4}$, y también sin los 284 valores exactamente 0 o 1
    (−0,0099).
  - **Lo que no está probado:** que la causa sea la asimetría a la izquierda, ni que esto explique por sí
    solo el peor RMSE de la red. La red Beta tiene un sesgo medio de −0,012 en validación y subestima sobre
    todo en los deciles altos (sección 6.3, apartado 6). Además, su RMSE mejora al aumentar $\varepsilon$
    (0,1799 con $10^{-4}$ y 0,1788 con $10^{-2}$; `results/tables/lgd_sensibilidad_eps.csv`), lo que apunta
    también al tratamiento de los valores cercanos a 1, no solo a la asimetría.
- **5. Corrección:** la frase precisa es "la MLE de la Beta iguala log-momentos y no la media; en estos
  datos, sin variables, la media ajustada queda 0,0095 por debajo de la real". Que eso explique el sesgo
  y el peor RMSE de la red es una hipótesis.
- **6. Lección:** separar lo que sale de una derivación de lo que es una explicación plausible. Antes de
  decir "esto explica", buscar un control (aquí, datos simulados de una Beta) y otras causas posibles
  (aquí, $\varepsilon$).

---

### Verificaciones de P3 sin hallazgo

Afirmaciones (de una IA o propias) que se comprobaron y resultaron correctas, más un problema de código
encontrado al verificar.

| Qué se afirmó o comprobó | Cómo se verificó | Resultado |
|---|---|---|
| La forma estable habitual de la BCE, `max(z,0) - z·y + log1p(exp(-\|z\|))`, tiene gradiente $(\sigma(z)-y)/n$ | autograd en float64 con $z = [0, 0]$, $y = [0, 1]$ | **Falla en $z = 0$**: autograd da $[0; -0{,}5]$ en vez de $[0{,}25; -0{,}25]$, porque `max` y `abs` no son derivables ahí. `src/models/losses.py` usa `logaddexp(0, z) - y·z`, que da el valor correcto (`tests/test_losses.py::test_gradiente_contra_autograd` incluye $z = 0$) |
| $\partial\,\text{BCE}/\partial z = (\sigma(z) - y)/n$ | autograd y `gradcheck` en float64 (`tests/test_losses.py`) | correcto |
| $\partial\,\text{MSE}/\partial z = 2(\sigma - y)\,\sigma(1-\sigma)/n$, y se anula con la sigmoide saturada | autograd, `gradcheck` y `test_mse_se_satura_con_logits_extremos` | correcto |
| $\int_0^1 \mu^y(1-\mu)^{1-y}\,dy \neq 1$: la BCE fraccional no es una verosimilitud | `scipy.integrate.quad` con $\mu$ = 0,5; 0,7; 0,9: 0,5000; 0,4721; 0,3641 | correcto |
| La BCE fraccional se minimiza en la media de $y$, con mínimo igual a la entropía $H(y)$ | `test_bce_fraccional_estima_la_media` y `test_bce_fraccional_minimo_en_y` | correcto |
| Adam: $t$ empieza en 1, $\varepsilon$ va fuera de la raíz, valores por defecto (0,9; 0,999; $10^{-8}$) | código fuente de PyTorch 2.14.1 y `test_adam_igual_a_pytorch` (50 pasos iguales) | correcto |
| El primer paso de Adam mide $\eta$; sin corrección de sesgo sería $\sqrt{10}\,\eta$ | `test_adam_primer_paso_mide_lr` | correcto |
| Adam es invariante a la escala de la pérdida y SGD no | `test_adam_invariante_a_la_escala_y_sgd_no` | correcto |
| SGD diverge en $f = (100x^2 + y^2)/2$ si $\eta > 2/100$ | ejecución: converge con 0,019 y diverge con 0,021; `test_cuadratica_mal_condicionada_sgd_diverge_y_adam_converge` | correcto |
| Revisión automática de Sourcery (otra IA) sobre el PR #9: carga insegura con joblib, NaN en `gradiente_beta` con logits extremos y carpeta de figuras inexistente | lectura del código | las tres observaciones son correctas o parcialmente correctas; ninguna es una alucinación |

**Tabla.** Verificaciones hechas por P3. *Fuente:* elaboración propia; pruebas en `tests/test_losses.py` y
`tests/test_optim.py`.
