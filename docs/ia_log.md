# Registro de uso de IA y cacería de alucinaciones

## Juan David Robayo (pérdidas, optimización y regularización)

Cada hallazgo sigue los seis puntos del enunciado. Las cifras se recalcularon con la partición aleatoria
vigente (`split_seed: 90`). Cuando la respuesta de la IA cita números, son los de la partición que había
en ese momento.

---

### JR-1. "Eso no equivale a sumar λ‖w‖²; esa es la diferencia entre Adam y AdamW" (razón equivocada)

- **Categoría:** matemáticas / conceptos (optimizador y regularización)
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-24
- **1. Prompt (literal):**
  > bien, parece que mis compañeros ya hicieron varios merge a develop, quieres verlo para que sepas
  > en que avance va el grupo?
- **2. Respuesta de la IA (parte relevante, literal):**
  > 7. **L2 implementada como `weight_decay` dentro de Adam.** Con Adam eso no equivale exactamente a
  > sumar λ‖w‖² a la pérdida; esa es la diferencia entre Adam y AdamW. Es un buen punto para tu
  > sección 3.5 y posible material para la cacería de alucinaciones.
- **3. Sospecha:** tomada al pie de la letra, la frase es cierta solo por un factor: el gradiente de
  $\tfrac{\lambda}{2}\lVert w\rVert^2$ es $\lambda w$, así que `weight_decay=λ` equivale a sumar
  $\tfrac{\lambda}{2}\lVert w\rVert^2$ y no $\lambda\lVert w\rVert^2$. Lo dudoso es la razón: atribuirlo a la
  diferencia entre Adam y AdamW sugiere que el `weight_decay` de Adam no es una L2. Si suma $\lambda w$
  al gradiente antes de los momentos, debería ser exactamente una L2.
- **4. Evidencia:**
  - En el código fuente de PyTorch 2.14.1 (`torch/optim/adam.py`), con `weight_decay` y sin
    `decoupled_weight_decay`, Adam hace `grad = grad.add(param, alpha=weight_decay)` antes de actualizar
    los momentos.
  - Experimento `tests/test_optim.py::test_weight_decay_de_adam_es_l2_y_adamw_no`: MLP 5→8→1 en float64,
    $\lambda = 0{,}1$, 30 pasos desde los mismos pesos. `Adam(weight_decay=0,1)` y Adam sobre
    `bce + 0,1 · penalizacion_l2` difieren en menos de $10^{-16}$ (iguales a precisión de
    máquina en cada paso). `AdamW` se separa en 0,58.
  - Comando: `python -m pytest tests/test_optim.py -k weight_decay`.
- **5. Corrección:** con `weight_decay=λ`, Adam minimiza exactamente $L + \tfrac{\lambda}{2}\lVert w\rVert^2$:
  el `weight_decay` acoplado **sí** es una L2 en la pérdida, y la única diferencia con "sumar λ‖w‖²" es el
  factor ½ (equivale a usar 2λ). Lo que no equivale a una L2 es el decaimiento desacoplado de AdamW. El
  único reparo real al uso de `weight_decay` en la red de PD es que también penaliza los sesgos
  (sección 3.5).
- **6. Lección:** cuando una IA dice "A no equivale a B", escribir las dos actualizaciones y
  comprobarlas con un experimento de pocos pasos antes de repetirlo en el reporte.

---

### JR-2. "La MSE tiende a encoger las predicciones hacia la media" (como defecto propio de la MSE)

- **Categoría:** conceptos
- **IA:** Claude (Claude Code, Opus 5.5) · **fecha:** 2026-10-24
- **1. Prompt (literal; entre corchetes, los nombres internos de los roles del equipo):**
  > Antes de escribir código, quiero confirmar que entendiste mi rol y acordar un plan.
  >
  > PARTE A. Confirmación de comprensión (con tus palabras, no copies la guía)
  > Soy [mi rol] (Optimización, regularización y pérdidas). Explícame:
  > 1. Qué tengo que entregar en la Parte 1 (PD) y en la Parte 2 (LGD).
  > 2. Qué artefactos son míos (módulos, configs, secciones del reporte, modelos exportados) y cuáles solo reviso o consumo.
  > 3. Qué contratos de la sección 7 de la guía me involucran (por ejemplo get_loss(name), build_mlp(cfg), load_splits, metrics.*) y quién produce o consume cada uno.
  > 4. Qué me toca en cada fase (0 a 10) y qué parte de la cacería de la alucinación y del video individual es mía.
  > 5. A quién reviso en el anillo y quién me revisa a mí.
  > Si algo de la guía es ambiguo o contradictorio, dímelo en vez de asumir.
  >
  > PARTE B. Estado real del repo
  > Con base en lo que ya leíste (estructura, git log, ramas remotas, src/), dime qué existe y qué falta de lo que yo necesito de [los demás integrantes]. Marca cada dependencia como "disponible", "parcial" o "bloqueada".
  >
  > PARTE C. Plan de trabajo para terminar mi parte con éxito
  > Propón un plan por etapas, en orden, donde cada etapa indique:
  > - Qué hago y qué archivos toca.
  > - De qué depende (y si puedo avanzar con datos sintéticos mientras espero).
  > - Cómo verifico que quedó bien (pruebas, verificación de gradientes contra autograd y diferencias finitas, comparación contra una implementación de referencia).
  >
  >
  > PARTE D. Decisiones abiertas
  > Lista las decisiones técnicas que debo tomar o acordar con el equipo antes de avanzar (framework, política de semillas, tratamiento de 0 y 1 en la Beta, pérdida candidata para LGD, presupuesto de búsqueda, etc.). Para cada una, da las opciones con sus pros y contras y tu recomendación, pero no decidas por mí.
  >
  > Reglas para esta conversación:
  > - No escribas ni modifiques archivos todavía. Solo responde.
  > - Explícame el porqué de cada paso, porque tengo que poder defender y explicar mi parte en la sustentación.
  > - Cuando empecemos a implementar, quiero ir paso a paso, una etapa por vez, y que me expliques cada decisión antes de aplicarla.
  > - No cites referencias bibliográficas de memoria; si hace falta una, márcala como "por verificar".
  > .no hacer comit sin mi permiso y me dices en que momento es recomendable hacerlo y tamvien dime como deberia capturar alucinaciones?
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

### JR-3. "La Beta con φ constante es exacta en los deciles bajos" (comparación mal planteada)

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

### JR-4. "El techo de la tanh subestimaría de forma sistemática los créditos de LGD alta"

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
  condicional (JR-2), común a todos los modelos.
- **6. Lección:** para afirmar que un modelo "subestima" un grupo hay que comparar medias predichas y
  observadas dentro de grupos definidos por la **predicción** (calibración por deciles), no mirar los
  valores individuales extremos.

---

### JR-5. "La Beta subestima por la asimetría a la izquierda; esto explica su peor RMSE" (causa sin evidencia)

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
  - **Control.** Con 41.173 valores simulados de la Beta ajustada (semilla 0), la media ajustada coincide
    con la muestral (diferencia +0,0003). Con la LGD real queda 0,0095 por debajo de la media (0,694 frente
    a 0,704), con
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

### Verificaciones sin hallazgo (Juan David Robayo)

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
| Referencias APA de 3.3, 3.5 y 5.1 entregadas por una IA como "verificadas" (2026-10-30; el prompt no se conservó, por eso no se registra como hallazgo) | Crossref (`api.crossref.org/works/<DOI>`), arXiv, JMLR y el registro del libro de MIT Press | **Errores encontrados:** para Papke y Wooldridge (1996) dio dos DOI distintos, y el de la forma `(199611/12)` no está en Crossref (el correcto termina en `(199611)11:6<619::AID-JAE418>3.0.CO;2-1`); para Platt dio 1999 y 2000 en la misma respuesta (el libro de MIT Press es de 2000); omitió el número 85 de scikit-learn en *JMLR* 12; citó ICLR como "Proceedings" aunque no tiene actas formales. Las otras seis entradas eran correctas. Se corrigieron en las secciones |

**Tabla.** Verificaciones hechas por Juan David Robayo. *Fuente:* elaboración propia; pruebas en `tests/test_losses.py` y
`tests/test_optim.py`.
