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
*(Esta subsección se completará al finalizar la evaluación del modelo en conjunto con P4 mediante análisis de coeficientes y valores SHAP).*
