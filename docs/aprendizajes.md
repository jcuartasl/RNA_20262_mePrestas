# Aprendizajes del proceso de modelamiento

## Juan David Robayo (pérdidas, optimización y regularización)

### 1. PD: el problema no fue discriminar, fue el nivel de la probabilidad

La logística de referencia reproduce exactamente la tasa de incumplimiento del periodo en que se entrenó
(PD media 0,1928 frente a 0,1928 en 2007–2013), pero subestima la de los años siguientes (0,215 frente a
0,262 en 2014; 0,221 frente a 0,255 en 2015). No es un error de ajuste: la tasa observada cambia de un año a
otro, en parte por la censura de los créditos recientes. La regularización casi no importó (con 163.541
créditos y 83 parámetros, toda la rejilla L2 con $C \geq 0{,}003$ da la BCE entre 0,54246 y 0,54263), y la
recalibración de Platt ajustada en 2014 corrigió el nivel de 2015 pero se pasó (0,268 frente a 0,255) sin
cambiar la AUC. **Lección:** en PD con partición temporal, la calibración es un blanco móvil; una
recalibración fija en un año no garantiza el nivel del siguiente, y hay que medirla en un periodo que no se
usó para ajustarla (secciones 5.1 y 3.5).

### 2. LGD: la pérdida importó poco; lo que limita es la información de las variables

Con la misma arquitectura, la BCE fraccional y la MSE empataron (RMSE 0,17728 y 0,17741 en 30 semillas) y
la Beta quedó 1,5 % peor; la red mejoró a la lineal solo 2,4 % de RMSE ($R^2$ de 0,192 a 0,231). Todas las
predicciones quedaron encogidas (desviación estándar 0,097 frente a 0,202 observada), y eso no es un defecto
de una pérdida: un modelo que predice la media condicional no puede reproducir la dispersión que las
variables no explican. Por eso la predicción prerregistrada de que la Beta reproduciría mejor la forma de la
distribución falló: describe mejor la LGD marginal, pero sus medias por crédito son las más encogidas.
**Lección:** antes de elegir una pérdida por la forma de $y$, preguntar qué se va a usar de la red; para
predicciones puntuales, la media condicional, y comparar distribuciones solo frente a lo que esa media
puede reproducir (secciones 3.3 y 6.3).

### 3. Transversal: los errores más costosos fueron los silenciosos

Varios errores del proyecto no produjeron ningún mensaje y dieron resultados plausibles:

- Tablas de evaluación que cruzaban predicciones de una partición aleatoria con las etiquetas de otra (la
  semilla de partición cambió y las predicciones guardadas no se regeneraron). El RMSE, 0,226, parecía solo
  un modelo malo; lo delató la pendiente de la calibración lineal, −0,04, cuando un modelo que predice
  medias debe dar cerca de 1.
- Un cambio de parámetro por línea de comandos con la clave incompleta (`lr=...` en vez de `train.lr=...`)
  se ignoraba sin aviso, y la demo entrenaba una red distinta de la final porque usaba valores por defecto.
- La BCE de la logística se calculaba en float32 y cambiaba en la cuarta cifra decimal por un solo crédito
  extremo.

**Lección:** una cifra plausible no prueba que el cálculo esté bien. Sirvieron tres verificaciones baratas:
contrastar una propiedad que debe cumplirse (pendiente ≈ 1, PD media = tasa de entrenamiento), reproducir la
cifra desde cero con el código y los datos actuales, y escribir pruebas sobre el comportamiento observable
(qué modelo guarda la demo) y no solo sobre funciones aisladas.
