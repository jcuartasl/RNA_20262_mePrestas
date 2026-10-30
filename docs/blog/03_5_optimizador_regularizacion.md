# 3.5 Optimizador y regularización

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  
**Estado:** borrador para integración

> La sección 3.3 dice **qué** se minimiza (la pérdida, derivada por máxima verosimilitud). Esta dice
> **cómo** se minimiza (SGD y Adam) y **cómo se evita el sobreajuste** (L1, L2, dropout y parada
> temprana). Cada propiedad que se afirma está respaldada por una prueba de `tests/test_optim.py`, que
> usa un Adam escrito a mano (`src/models/optim.py`) y lo compara con el de PyTorch. La numeración de
> tablas (3.5.x) y las remisiones a otras secciones, que siguen la estructura de la guía del curso, son provisionales hasta la integración del blog.

---

## 1. Descenso de gradiente y SGD

El gradiente $g_t = \nabla_\theta L(\theta_{t-1})$ apunta hacia donde la pérdida sube más rápido. El
descenso de gradiente da un paso de tamaño $\eta$ (la tasa de aprendizaje) en la dirección contraria:

$$\theta_t = \theta_{t-1} - \eta\, g_t.$$

Con 163.541 créditos de PD o 41.173 de LGD, calcular el gradiente con todos los datos en cada paso es
caro. El **descenso de gradiente estocástico (SGD)** usa el gradiente de un lote aleatorio (256 créditos
en la red de LGD, 512 en la de PD). Es un estimador ruidoso pero insesgado del gradiente completo, y
permite muchos pasos por época. Una **época** es una pasada completa por el entrenamiento; los lotes se
barajan en cada época con un generador con semilla.

SGD tiene tres problemas que motivan Adam:

1. **Una sola $\eta$ para todos los parámetros.** Si unos tienen gradientes grandes y otros pequeños,
   ninguna $\eta$ sirve para todos.
2. **El ruido del lote** hace zigzaguear la trayectoria.
3. **Depende de la escala de la pérdida.** Multiplicar la pérdida por 100 multiplica el paso por 100.

## 2. Adam

Adam (Kingma y Ba, 2015) combina dos promedios móviles exponenciales del gradiente, elemento a elemento:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2.$$

- $m_t$ es un **momento**: suaviza el ruido del lote y acelera en las direcciones consistentes. Con
  $\beta_1 = 0{,}9$ recuerda del orden de 10 pasos.
- $\sqrt{v_t}$ estima el **tamaño típico del gradiente** de cada parámetro. Dividir por él da a cada
  parámetro su propia tasa de aprendizaje efectiva.

**Corrección de sesgo.** Los dos promedios arrancan en $m_0 = v_0 = 0$, así que al principio están
sesgados hacia cero. Desarrollando la recursión,

$$m_t = (1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k} g_k \quad\Rightarrow\quad
\mathbb{E}[m_t] = \mathbb{E}[g]\,(1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k} = \mathbb{E}[g]\,(1-\beta_1^t),$$

si el gradiente tiene media constante (la suma geométrica vale $(1-\beta_1^t)/(1-\beta_1)$). Lo mismo
pasa con $v_t$ y $\beta_2$. Se corrige dividiendo:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}.$$

**Actualización:**

$$\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}, \qquad
\beta_1 = 0{,}9,\quad \beta_2 = 0{,}999,\quad \varepsilon = 10^{-8}.$$

Son los valores por defecto de `torch.optim.Adam` (comprobado en PyTorch 2.14.1). El contador $t$ empieza
en 1 en el primer paso (con $t = 0$ la corrección dividiría por cero), y $\varepsilon$ va fuera de la raíz.

La implementación propia sigue las ecuaciones línea a línea:

```python
estado["t"] += 1
t = estado["t"]
with torch.no_grad():
    for p, m, v in zip(params, estado["m"], estado["v"]):
        m.mul_(b1).add_((1 - b1) * p.grad)                     # m_t
        v.mul_(b2).add_((1 - b2) * p.grad ** 2)                # v_t
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)    # corrección de sesgo
        p -= lr * m_hat / (v_hat.sqrt() + eps)                 # actualización
```

*Fragmento de `paso_adam` en `src/models/optim.py`.*

| Propiedad | Por qué se cumple | Prueba que la respalda |
|---|---|---|
| El Adam propio es el de PyTorch | Mismas ecuaciones y mismo orden de operaciones | `test_adam_igual_a_pytorch`: mismos parámetros en **cada uno** de 50 pasos (float64) |
| El primer paso mide $\eta$ | En $t = 1$, $\hat m_1 = g_1$ y $\hat v_1 = g_1^2$: el paso es $\eta\, g_1/\lvert g_1\rvert$ | `test_adam_primer_paso_mide_lr`: paso de exactamente 0,01 con gradientes de 0,004 y de 100. Sin corrección de sesgo sería $\sqrt{10}\,\eta \approx 3{,}16\,\eta$ |
| Invariancia a la escala | Si la pérdida se multiplica por $c$, $\hat m$ y $\sqrt{\hat v}$ también, y el cociente no cambia | `test_adam_invariante_a_la_escala_y_sgd_no`: pérdida × 1000 deja igual la trayectoria de Adam; con SGD los pasos crecen más de 100 veces |
| Adam tolera el mal condicionamiento | Reescala cada eje por separado (apartado 3) | `test_cuadratica_mal_condicionada_sgd_diverge_y_adam_converge` |

**Tabla 3.5.1.** Propiedades de Adam y la prueba que respalda cada una. *Fuente:* elaboración propia;
`src/models/optim.py` y `tests/test_optim.py` (7 pruebas, todas pasan).

La invariancia a la escala da una lectura práctica de $\eta$: con Adam es, aproximadamente, cuánto se
mueve como máximo cada parámetro por paso, sin importar en qué unidades esté la pérdida.

## 3. Por qué Adam y no SGD

En la función $f(x, y) = \tfrac12 (100x^2 + y^2)$, la curvatura en $x$ es 100 veces la de $y$. SGD hace en
la dirección $x$ la actualización $x \leftarrow (1 - 100\eta)\,x$, que solo converge si
$\lvert 1 - 100\eta\rvert < 1$, es decir, si $\eta < 2/100 = 0{,}02$. Con un $\eta$ tan pequeño, la
dirección $y$ avanza muy despacio. Con $\eta = 0{,}025$, SGD diverge y Adam converge a menos de $10^{-6}$
(Tabla 3.5.1). Las redes de este trabajo tienen miles de parámetros con escalas de gradiente muy distintas,
así que ambas usan Adam.

## 4. L2 y L1: regularización como a priori

Una red con 20.993 parámetros puede memorizar el entrenamiento. La regularización añade a la pérdida un
término que castiga los pesos grandes:

$$L_{\text{reg}}(\theta) = L(\theta) + \lambda\,\Omega(\mathbf{w}).$$

**L2.** $\Omega_{L2}(\mathbf{w}) = \tfrac12 \sum_j w_j^2$, con gradiente $\mathbf{w}$. Tiene la misma
lectura probabilística que las pérdidas de la sección 3.3: si a la verosimilitud se le suma una a priori
gaussiana $w_j \sim \mathcal{N}(0, \tau^2)$, maximizar la probabilidad a posteriori (MAP) equivale a
minimizar

$$-\log p(\mathbf{y}\mid\mathbf{w}) + \sum_j \frac{w_j^2}{2\tau^2}.$$

Con la pérdida promediada entre los $n$ datos queda $\lambda = 1/(n\tau^2)$: una a priori más estrecha
es un $\lambda$ mayor. La pérdida sale de la verosimilitud y la regularización, de la a priori. La prueba
`test_l2_encoge_los_pesos` lo comprueba: en una logística entrenada con el Adam propio, la norma de los
pesos baja estrictamente con $\lambda = 0;\ 0{,}01;\ 0{,}1;\ 1$.

Con SGD, la L2 da $w \leftarrow (1 - \eta\lambda)\, w - \eta g$: el peso se encoge multiplicativamente en
cada paso. Por eso se le llama también *decaimiento de pesos* (*weight decay*). Con Adam, las dos cosas
dejan de ser iguales (apartado 5).

**L1.** $\Omega_{L1}(\mathbf{w}) = \sum_j \lvert w_j\rvert$, con subgradiente $\text{signo}(\mathbf{w})$.
Corresponde a una a priori de Laplace, $p(w_j) \propto e^{-\lvert w_j\rvert / b}$, con un pico agudo en 0.
Por eso **produce esparsidad**: el óptimo tiende a caer en puntos donde algunos coeficientes valen
exactamente 0, y la L1 selecciona variables. En la logística de PD (sección 5.1), la L1 con $C = 0{,}01$
conserva 34 de las 82 variables y pierde solo 0,0004 de AUC.

Los ceros exactos salen de métodos especializados, como los pasos proximales de `saga` en scikit-learn.
`saga` da un paso de gradiente sobre la pérdida, $u = w - \eta\, g$, y luego aplica el operador proximal
de la L1, el umbral suave:

$$w_j \leftarrow \operatorname{signo}(u_j)\,\max\big(\lvert u_j\rvert - \eta\lambda,\ 0\big).$$

Todo coeficiente con $\lvert u_j\rvert \le \eta\lambda$ queda exactamente en 0. Si la L1 se suma a la
pérdida y se entrena con SGD o Adam, no existe ese paso: los pesos irrelevantes oscilan alrededor de 0 y
casi nunca valen exactamente 0.

**Los sesgos no se penalizan.** El sesgo de salida fija el nivel base (la LGD media es cerca de 0,70).
Penalizarlo lo empujaría hacia 0, es decir, hacia una predicción de 0,5, sin ninguna razón. La a priori
"pesos pequeños" tiene sentido para los coeficientes de las variables, no para el nivel base. Por eso
`penalizacion_l1` y `penalizacion_l2` (`src/models/losses.py`) solo suman sobre los pesos, y la
logística de scikit-learn tampoco penaliza el intercepto: si lo hiciera, la PD media predicha en
entrenamiento no coincidiría con la tasa observada, y coincide (sección 5.1, apartado 4).

## 5. L2 con Adam frente a AdamW

`torch.optim.Adam(weight_decay=λ)` suma $\lambda\mathbf{w}$ al gradiente **antes** de calcular $m_t$ y
$v_t$. Eso es exactamente minimizar $L + \tfrac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ con Adam. Como el
término $\lambda\mathbf{w}$ pasa por la división entre $\sqrt{\hat v_t}$, los pesos con gradientes grandes
reciben menos regularización efectiva.

**AdamW** (Loshchilov y Hutter, 2019) desacopla el decaimiento: aplica
$\mathbf{w} \leftarrow \mathbf{w} - \eta\lambda\mathbf{w} - \eta\,\hat m_t / (\sqrt{\hat v_t} + \varepsilon)$,
con el término $\lambda\mathbf{w}$ fuera de los momentos. **No equivale a una L2 en la pérdida.**

La prueba `test_weight_decay_de_adam_es_l2_y_adamw_no` lo comprueba en un MLP 5→8→1 con $\lambda = 0{,}1$
durante 30 pasos:
- `Adam(weight_decay)` y Adam sobre `bce + 0,1 · penalizacion_l2` dan los mismos parámetros en cada paso
  (diferencia máxima menor que $10^{-16}$, error de redondeo);
- AdamW se separa (diferencia de 0,58).

Esta distinción importa para este trabajo, porque la red de PD usa `weight_decay` y la de LGD usa la
penalización explícita (apartado 8). Las dos son L2.

## 6. Dropout

Con dropout (Srivastava et al., 2014), en entrenamiento cada neurona oculta se apaga con probabilidad $p$ y las que quedan se reescalan:

$$\tilde h_j = \frac{r_j\, h_j}{1-p}, \qquad r_j \sim \text{Bernoulli}(1-p).$$

El factor $1/(1-p)$ (*inverted dropout*, el de `nn.Dropout` de PyTorch) mantiene
$\mathbb{E}[\tilde h_j] = h_j$, así que al predecir no se apaga nada ni se reescala. La red no puede
depender de unas pocas neuronas, y el efecto se parece a promediar muchas subredes.

En la práctica, esto obliga a cambiar de modo: `modelo.train()` activa el dropout y `modelo.eval()` lo
apaga. Si se olvida `eval()` al validar o predecir, las predicciones salen ruidosas y cambian en cada
llamada. Por eso `entrenar_lgd_mlp` y `predecir_lgd` llaman a `eval()` antes de evaluar.

## 7. Parada temprana

Después de cada época se mide la pérdida en validación. Si mejora, se guardan los pesos. Si no mejora
durante `patience` épocas seguidas, el entrenamiento se detiene y **se restauran los pesos de la mejor
época**, no los de la última. Regulariza porque limita cuánto crecen y se especializan los pesos.

Formalmente, sea $\theta_e$ el vector de pesos al terminar la época $e$ y $L_{\text{val}}(e)$ su pérdida
de validación. Tras cada época se actualiza $e^* = \arg\min_{e' \le e} L_{\text{val}}(e')$ (solo cuenta una
mejora estricta). El entrenamiento se detiene en la primera época $E$ con $E - e^* \ge \text{patience}$, o
al llegar al máximo de épocas, y se devuelven $\hat\theta = \theta_{e^*}$ (`entrenar_lgd_mlp` en
`src/models/lgd_mlp.py`).

En la búsqueda de LGD (sección 6.3), 31 de las 36 configuraciones pararon antes del máximo de 100 épocas.
En promedio entrenaron 46,1 épocas y la mejor fue la 37,1.

Tiene un costo: la validación se usa para decidir cuándo parar y para elegir la configuración, así que su
métrica queda algo optimista. Por eso la prueba se reserva para la evaluación final.

## 8. Qué usa cada modelo

La Tabla 3.5.2 resume el optimizador y la regularización de la configuración final de cada modelo.

| | Logística de PD | MLP de PD | MLP de LGD |
|---|---|---|---|
| Optimizador | `lbfgs` (L2) o `saga` (L1), scikit-learn | Adam, lr 0,003 | Adam, lr 0,003 |
| Lote | todos los datos | 512 | 256 |
| L2 | $C$ buscado; elegido $C = 0{,}01$ | `weight_decay` = $10^{-4}$, fijo | penalización explícita, $\lambda = 10^{-5}$ (buscado) |
| ¿Penaliza los sesgos? | no | **sí** (`weight_decay` sobre todos los parámetros) | no |
| L1 | buscada; no elegida | no | disponible; no usada |
| Dropout | — | 0 (buscado entre 0; 0,2 y 0,3) | 0,1 (buscado) |
| Parada temprana | — (converge por tolerancia) | paciencia 5, máximo 30 épocas | paciencia 10, máximo 100 épocas |

**Tabla 3.5.2.** Optimizador y regularización de la configuración final de cada modelo. *Fuente:*
elaboración propia; `configs/pd_logreg.yaml`, `configs/pd_mlp.yaml`, `configs/lgd_mlp.yaml`,
`src/models/pd_mlp.py`, `src/models/lgd_mlp.py` y `src/search/pd_random_search.py`.

Las dos redes regularizan los sesgos de forma distinta. Con $\lambda = 10^{-4}$ el efecto sobre los sesgos
de la red de PD probablemente es pequeño, pero no se midió.

Un detalle de la red de LGD: la pérdida de entrenamiento que se guarda en el historial incluye la
penalización L2, y la de validación no. Con los pesos finales de la semilla 0, ese término vale
$10^{-5} \times 175{,}2 = 0{,}0018$. Hay que tenerlo en cuenta al comparar las dos curvas (sección 6.3,
Figura 6.3.3).

## 9. Qué dicen los datos sobre la regularización

La Tabla 3.5.3 agrupa los 36 entrenamientos de la búsqueda de LGD según su L2 y su dropout.

| Hiperparámetro | Valor | Entrenamientos | Mediana del RMSE | Mínimo del RMSE |
|---|---|---|---|---|
| L2 ($\lambda$) | 0 | 12 | 0,17900 | 0,17833 |
| | $10^{-5}$ | 12 | 0,17941 | **0,17735** |
| | $10^{-4}$ | 9 | 0,18009 | 0,17756 |
| | $10^{-3}$ | 3 | 0,18079 | 0,18075 |
| Dropout | 0 | 12 | 0,17917 | 0,17756 |
| | 0,1 | 9 | 0,17991 | **0,17735** |
| | 0,2 | 6 | 0,17899 | 0,17842 |
| | 0,3 | 9 | 0,18079 | 0,17833 |

**Tabla 3.5.3.** RMSE en validación de la búsqueda de LGD según la L2 y el dropout (36 entrenamientos:
12 configuraciones × 3 pérdidas). *Fuente:* elaboración propia; `results/tables/lgd_busqueda.csv`.

- **La mediana empeora a medida que crece la L2.** Es un indicio de que, con 41.173 créditos y una red de
  20.993 parámetros, el sobreajuste no es el problema principal. La parada temprana ya lo controla.
- **El dropout no muestra un patrón claro.**
- **Esta tabla no aísla el efecto de cada hiperparámetro.** Con 12 configuraciones aleatorias los
  hiperparámetros están mezclados (por ejemplo, $\lambda = 10^{-3}$ aparece en una sola configuración), y
  cada nivel incluye las tres pérdidas. Medir el efecto de cada uno exigiría variarlo con los demás fijos.

La logística de PD apunta a lo mismo: con 163.541 créditos y 83 parámetros, toda la rejilla de L2 con
$C \geq 0{,}003$ da la BCE de validación entre 0,54246 y 0,54263 (sección 5.1).

## 10. Limitaciones

- La comparación entre SGD y Adam se hizo en problemas de juguete (una logística y una cuadrática), no en
  las redes del trabajo. Las redes se entrenaron solo con Adam.
- No se buscaron $\beta_1$, $\beta_2$ ni $\varepsilon$; se usaron los valores por defecto.
- No se usó AdamW. Su diferencia con la L2 se mostró en una prueba, no en los modelos finales.
- El efecto de la L2 y del dropout en los datos reales es solo un indicio (apartado 9).

---

## Referencias

- Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization* [Conference paper]. 3rd
  International Conference on Learning Representations, San Diego, CA, United States.
  https://doi.org/10.48550/arXiv.1412.6980
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization* [Conference paper].
  International Conference on Learning Representations (ICLR 2019). https://doi.org/10.48550/arXiv.1711.05101
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple
  way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*(56),
  1929–1958. https://jmlr.org/papers/v15/srivastava14a.html
