# Hipótesis de Negocio para el Análisis Exploratorio y Modelado

**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Universidad:** Universidad Nacional de Colombia  
**Proyecto:** Trabajo 2 — Redes neuronales para datos tabulares (*¿me prestas?*)  
**Responsable:** Persona 1 (Datos, objetivos PD/LGD y pérdida esperada)  

---

## Hipótesis Explícitas de Riesgo

1. **H1 — Sobreendeudamiento (DTI):**  
   *Hipótesis:* Solicitantes con ratios DTI superiores al percentil 75 presentarán una probabilidad de incumplimiento ($PD$) significativamente mayor que aquellos con DTI bajo control ($<15\%$).  
   *Mecanismo económico:* Menor capacidad de absorción ante pérdidas imprevistas de ingresos o incrementos en gastos fijos.

2. **H2 — Búsqueda de Liquidez Reciente (`inq_last_6mths`):**  
   *Hipótesis:* Los clientes con 3 o más indagaciones en los últimos 6 meses exhibirán una tasa de default al menos 1.5 veces superior a los solicitantes sin indagaciones recientes.  
   *Mecanismo económico:* Señal adversa en burós de crédito indicativa de sobreendeudamiento acelerado o rechazos previos en otras entidades.

3. **H3 — Utilización del Crédito Rotativo (`revol_util`):**  
   *Hipótesis:* Un porcentaje de utilización rotativa cercano a su límite ($>80\%$) es uno de los predictores continuos no lineales más fuertes de mora severa.  
   *Mecanismo económico:* Agotamiento de líneas de emergencia crediticia.

4. **H4 — Estabilidad y Antigüedad Laboral (`emp_length`):**  
   *Hipótesis:* Prestatarios con antigüedad menor a 1 año presentan mayor dispersión y mayor propensión al incumplimiento temprano frente a empleados con más de 10 años de trayectoria.  
   *Mecanismo económico:* Menor estabilidad en el flujo de caja salarial.

5. **H5 — Severidad de la Pérdida según Monto Financiado (`funded_amnt`):**  
   *Hipótesis:* En créditos en default, préstamos de gran cuantía ($>\$25,000\text{ USD}$) reportan una LGD media significativamente mayor que créditos pequeños ($<\$5,000\text{ USD}$).  
   *Mecanismo económico:* Mayor dificultad de cobro extrajudicial integral sobre obligaciones de alto volumen no garantizadas.

---

### Registro de Contraste y Cierre
*Redactado por P4 a partir de `python -m src.experiments.contrastar_hipotesis` (`results/tables/p4_hipotesis_contraste.csv`, `results/tables/p4_hipotesis_revol_util_deciles.csv`) y de la importancia por permutación de la regresión logística en prueba temporal (`python -m src.experiments.interpret_pd`, `results/tables/p4_importancia_pd_logreg_test_temporal.csv`). Pendiente de revisión por P1.*

**Método.** Pruebas unilaterales con α = 0,05: diferencia de proporciones (z) e intervalo de confianza del 95 % de la razón de tasas (log-RR) para H1–H4; t de Welch para H5; razón de verosimilitud entre un logit lineal y uno cúbico en `revol_util` para la no linealidad de H3. Muestras: entrenamiento temporal (originaciones 2007–2013) y prueba temporal (2015). Con decenas de miles de créditos casi cualquier diferencia es significativa, por lo que el veredicto se apoya en el tamaño del efecto y en su estabilidad entre ambas muestras.

| Hip. | Veredicto | Evidencia descriptiva: train / test | Evidencia del modelo (caída de AUC al permutar) |
|---|---|---|---|
| H1 | Confirmada | Tasa 24,9 % vs. 15,8 % (razón 1,58; IC 1,54–1,61) / 36,2 % vs. 19,1 % (razón 1,90; IC 1,81–1,99) | 4.ª variable (0,018); a mayor DTI, mayor riesgo |
| H2 | No confirmada en la magnitud planteada | Razón 1,46 (IC 1,41–1,50) / 1,30 (IC 1,21–1,39): más riesgo, pero por debajo del umbral de 1,5 | 7.ª (0,008); dirección positiva |
| H3 | Parcialmente confirmada | `revol_util` > 80 %: razón 1,33 / 1,49. La tasa crece de forma casi monótona entre deciles (11,6 % a 24,9 %), con los mayores saltos en utilizaciones bajas y sin aceleración cerca del límite. No linealidad: LR = 26,1 (p = 2·10⁻⁶) en train, con efecto pequeño; no significativa en test (p = 0,26) | 2.ª (0,028), solo detrás del plazo |
| H4 | No confirmada (inestable) | Train: 18,5 % vs. 19,7 % (razón 0,94; IC 0,90–0,98), dirección contraria. Test: 29,2 % vs. 21,7 % (razón 1,34; IC 1,25–1,44) | 15.ª (0,003), entre las menos importantes |
| H5 | Confirmada | LGD media 0,676 vs. 0,582 (diferencia 0,094; IC 0,083–0,104) / 0,921 vs. 0,884 (diferencia 0,037; IC 0,027–0,047) | Débil: con la red de LGD, `monto_total` aumenta el RMSE en solo 0,0012 al permutarse (0,0003 en el lineal), frente a 0,026 del plazo (`term`), la variable dominante |

**Notas.**

- H3: se confirma que `revol_util` es de los predictores más fuertes, pero no la forma no lineal con umbral cerca del límite de crédito. La relación es aproximadamente lineal, incluso algo cóncava.
- H4: el efecto solo aparece en 2015, donde los créditos ya resueltos están sesgados hacia incumplimientos tempranos (sesgo de madurez). Esto es compatible con el mecanismo de incumplimiento temprano, pero no con un mayor riesgo a lo largo de la vida del crédito. La parte de "mayor dispersión" no se evaluó.
- Los contrastes descriptivos son marginales: comparan una variable a la vez sin controlar por las demás (por ejemplo, el DTI se relaciona con el plazo y el propósito). La importancia por permutación aporta la visión condicional dentro del modelo.
- En la importancia por permutación, `loan_amnt`, `funded_amnt` y `funded_amnt_inv` se permutan juntas: permutadas por separado, el modelo extrapola a combinaciones imposibles y su importancia se infla artificialmente.
- La LGD media de 2015 (≈0,9) refleja recuperaciones aún incompletas (censura), limitación ya descrita en la construcción de la LGD.
- Con la red neuronal de PD (P2), `revol_util` pasa a ser la variable más importante (caída de AUC 0,041), seguida del plazo y del ingreso; refuerza la parte de H3 sobre la fuerza del predictor. `emp_length` (H4) sigue fuera de las variables importantes (`results/tables/p4_importancia_pd_mlp_test_temporal.csv`).
- H5: el efecto del monto sobre la LGD es claro de forma marginal, pero pequeño una vez que el modelo considera el plazo y las demás variables (`results/tables/p4_importancia_lgd_mlp_bce_seed0_test_random.csv`).
