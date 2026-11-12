# Expected Credit Loss — Post-Challenge Analysis

Estimación de **Pérdida Esperada (ECL)** para una cartera de más de 2 millones de préstamos personales mediante:

$$ ECL_i = PD_i \times LGD_i \times EAD_i $$

Este repositorio documenta mi solución al **Reto 01 de Data Science**, el análisis posterior al feedback recibido y varias iteraciones para entender no solo cómo mejorar los modelos, sino también cómo tomar una mejor decisión con sus predicciones.

El objetivo del post-challenge no fue reconstruir retrospectivamente una solución ganadora. Quise identificar qué parte de mi primera propuesta estaba generando el exceso de ECL, corregirla de forma trazable y dejar planteados challengers para una posible iteración futura.

---

## 1. Problema

El reto requería estimar la pérdida esperada individual de cada crédito mediante tres componentes:

- **PD — Probability of Default:** probabilidad de incumplimiento.
- **LGD — Loss Given Default:** proporción de pérdida en caso de default.
- **EAD — Exposure at Default:** exposición al momento del incumplimiento.

Además del error predictivo, existían dos restricciones:

1. El **ECL total estimado debía ser superior a la pérdida real total**.
2. Como máximo el **10% de los préstamos** podía tener un ECL inferior a su pérdida real.

Por lo tanto, el problema no terminaba al estimar correctamente PD, LGD y EAD. También era necesario definir una política conservadora que cumpliera las restricciones sin elevar innecesariamente la pérdida esperada de toda la cartera.

---

## 2. Mi solución original

Mi primera solución terminó en **5.º lugar**.

El pipeline incluía:

- EDA y análisis de calidad de datos.
- Control de leakage y separación Train/Test.
- Ingeniería de variables basada en los hallazgos del EDA.
- PD mediante **WoE + regresión logística**.
- EAD y LGD mediante modelos separados sobre defaults.
- Integración:

$$ ECL = PD \times LGD \times EAD $$

- Política conservadora para cumplir la restricción individual.

Uno de los resultados más positivos fue la calibración de PD: la probabilidad media estimada quedó prácticamente igual a la tasa real de default.

El principal problema apareció después.

El **ECL basal estaba muy cerca de la pérdida real agregada**, pero aproximadamente 12.8% de las operaciones quedaba por debajo de su pérdida observada.

Para corregirlo, mi primera estrategia asignaba un piso conservador al grupo de mayor PD utilizando una proporción alta del importe financiado.

La política cumplía las reglas, pero era costosa:

$$ ECL / pérdida\ real \approx 1.90x $$

El feedback posterior fue bastante claro: el principal margen de mejora no estaba necesariamente en rehacer PD, LGD o EAD, sino en **cómo estaba distribuyendo el margen conservador**.

---

# 3. Post-Challenge V2

La segunda versión parte de una pregunta diferente:

> Si el ECL basal ya reproduce razonablemente la pérdida agregada, ¿cuánto necesita realmente cada operación para quedar cubierta?

En lugar de utilizar el importe financiado como referencia, utilizo primero la pérdida condicionada al default:

$$ LGD_i \times EAD_i $$

Pero todavía hay una segunda consideración: cada préstamo ya tiene un ECL basal.

Por eso el costo relevante no es toda la pérdida condicional, sino el **costo incremental de cobertura**:

$$
Costo_i =
m(LGD_iEAD_i)
-
PD_iLGD_iEAD_i
$$

Con esto construyo un criterio de priorización basado en:

$$
Score_i =
\frac{PD_i}
{Costo_i}
$$

La interpretación es simple: priorizo operaciones con mayor probabilidad de default por cada dólar adicional de ECL que necesito agregar.

---

## 4. Validación del margen

No quería que la mejora dependiera de ajustar manualmente la política hasta quedar justo debajo del límite del 10%.

Por eso:

- la política se aprende únicamente dentro de Train/Validación;
- la muestra de validación se divide para separar calibración y evaluación;
- se utilizan **300 remuestras bootstrap** para analizar la estabilidad del porcentaje de operaciones por debajo;
- Test permanece cerrado hasta que los modelos y la política están congelados.

Esto cambia la forma de interpretar el margen. Ya no busco simplemente obtener 9.99%, sino construir una política que conserve cierto colchón ante variaciones de la muestra.

---

# 5. Resultado principal — V2A

La **V2A** es la estrategia que considero principal dentro de este post-challenge.

No reduce artificialmente el ECL basal de las operaciones no cubiertas. La mejora proviene exclusivamente de:

- utilizar `LGD × EAD` como referencia de pérdida;
- medir el costo incremental;
- cambiar el ranking de cobertura;
- validar la política con bootstrap.

### Resultado

| Estrategia | ECL / pérdida real | % por debajo |
|---|---:|---:|
| Solución original | ~1.90x | ~9.8% |
| **V2A** | **1.6255x** | **9.49%** |

La mejora más importante del proyecto no vino de agregar más variables o cambiar inmediatamente de algoritmo, sino de formular mejor la decisión posterior al modelo.

---

# 6. Experimento adicional — κ

Después del reto revisé las soluciones compartidas por otros participantes y la solución de referencia.

Una estrategia utilizada en varias de las mejores soluciones consiste en aplicar un factor:

$$
\kappa < 1
$$

al ECL basal del segmento que no recibe cobertura.

Esto funciona particularmente bien bajo las reglas del reto porque los créditos sanos tienen pérdida observada igual a cero.

Probé esta lógica como **experimento posterior**, no como parte de mi solución original.

El resultado se acerca mucho al mínimo permitido por la función objetivo:

- ECL ≈ **USD 573 M**
- ECL / pérdida real ≈ **1.03x**
- operaciones por debajo ≈ **9.58%**

No considero esta estrategia directamente trasladable a una política bancaria de provisiones.

En un entorno real tendría que analizarse bajo criterios adicionales de:

- prudencia;
- estabilidad;
- regulación;
- políticas contables;
- validación independiente;
- gobernanza del modelo.

Por eso en este repositorio **V2A permanece como la propuesta principal** y κ se conserva como aprendizaje sobre la función objetivo del reto.

---

# 7. Challengers experimentales

Después de cerrar V2 dejé dos preguntas adicionales.

Estas pruebas no sustituyen el resultado principal. Funcionan como una base para una futura iteración del modelo.

---

## V3 — LightGBM como challenger de PD

La primera prueba consiste en cambiar únicamente el modelo de PD.

La regresión logística original obtuvo aproximadamente:

$$
AUC \approx 0.741
$$

El challenger LightGBM aumentó la discriminación a aproximadamente:

$$
AUC \approx 0.753
$$

sin deteriorar de forma material la calibración agregada.

Sin embargo, apareció un resultado importante:

> Utilizar una PD con mayor AUC y mantener la política de margen anterior no mejora automáticamente el ECL.

Fue necesario volver a calibrar la política utilizando la nueva PD.

Después de hacerlo:

| Estrategia | AUC PD | ECL total | ECL / pérdida | % debajo |
|---|---:|---:|---:|---:|
| V2A · logística | ~0.741 | USD 902.1 M | 1.6255x | 9.49% |
| **V3 · LightGBM** | **~0.753** | **USD 884.7 M** | **1.5942x** | **9.46%** |

La mejora equivale aproximadamente a:

$$
USD\ 17.4M
$$

o cerca de **1.9% menos ECL** respecto a V2A.

La conclusión no es simplemente que LightGBM sea mejor. El aprendizaje es que **modelo y política de decisión deben evaluarse como un sistema**.

---

## V4 — Regresión cuantílica de la pérdida

La segunda prueba cuestiona el piso utilizado para cubrir los créditos.

V2 y V3 utilizan una pérdida media condicionada al default:

$$
LGD_i \times EAD_i
$$

y aplican un multiplicador común.

Pero dos préstamos con la misma pérdida esperada no necesariamente tienen la misma incertidumbre.

Por eso probé modelar directamente:

$$
Q_{\alpha,i}
$$

es decir, un **cuantil individual de la pérdida**.

La política potencial sería:

\[
ECL_i =
\max(
ECL_i^{base},
factor \times Q_{\alpha,i}
)
\]

Realicé un piloto reproducible únicamente dentro de desarrollo, sin utilizar Test para seleccionar la estrategia.

### Resultado del piloto

| Estrategia | ECL validación | Ratio | % debajo | MAE | MAPE |
|---|---:|---:|---:|---:|---:|
| V3 · piso medio | USD 126.40 M | 1.6400x | 9.34% | USD 2,563 | 87.24% |
| **V4 · cuantil** | **USD 125.64 M** | **1.6301x** | **9.33%** | **USD 2,551** | **86.52%** |

La mejora es pequeña, alrededor de **0.6%**, pero consistente en varias métricas.

No extrapolo este resultado al Test oficial.

Lo interpreto únicamente como evidencia suficiente para justificar una V4 completa si el proyecto continuara.

---

# 8. Evolución del proyecto

El proyecto terminó mostrando tres niveles distintos de mejora:

### V1 — Modelar el riesgo

\[
PD \times LGD \times EAD
\]

El modelo basal ya estaba razonablemente calibrado a nivel agregado.

### V2 — Mejorar la decisión

La principal mejora vino de entender:

> ¿Dónde conviene gastar el margen conservador?

Se incorporó costo incremental y validación mediante bootstrap.

### V3 — Mejorar el ranking

LightGBM permitió identificar mejor dónde se concentra el riesgo y redujo adicionalmente el costo de margen.

### V4 — Mejorar cuánto cubrir

La regresión cuantílica introduce una forma de adaptar el piso a la incertidumbre individual de la pérdida.

---

# 9. Principales aprendizajes

Este reto me dejó varias ideas que quiero conservar para futuros proyectos de Data Science.

### 1. Un buen modelo no garantiza una buena decisión

PD, LGD y EAD pueden estar razonablemente calibrados y aun así la política posterior puede ser ineficiente.

### 2. La función objetivo importa

Optimizar una métrica requiere entender exactamente qué premia y qué penaliza.

Eso no significa que la estrategia óptima para una competencia sea necesariamente la misma que implementaría en producción.

### 3. El costo de actuar también forma parte del problema

No basta con identificar quién tiene mayor riesgo.

También importa cuánto cuesta actuar sobre ese riesgo.

### 4. Más AUC no implica automáticamente mejor resultado económico

V3 mostró que mejorar el ranking de PD solo agrega valor cuando la política posterior se vuelve a calibrar.

### 5. La incertidumbre también se puede modelar

La regresión cuantílica permite ir más allá de la pérdida media y estudiar directamente diferentes niveles de severidad.

### 6. El EDA debe terminar en decisiones

No utilizo el análisis exploratorio únicamente para describir variables. Cada transformación, exclusión o variable derivada debe responder a un hallazgo previo.

### 7. Revisar otras soluciones también es parte del aprendizaje

Este repositorio distingue explícitamente:

- qué desarrollé durante el reto;
- qué cambié después del feedback;
- qué aprendí posteriormente revisando otras soluciones.

Esa trazabilidad me parece más valiosa que intentar presentar retrospectivamente todas las mejoras como parte de la solución original.

---

# 10. Estructura del proyecto

```text
.
├── Reto1_ECL_PostChallenge_Portafolio_Experimentos_LM.ipynb
├── README.md
└── Data
   ├── X.csv
   └── target.csv
```

Los archivos `X.csv` y `target.csv` deben colocarse en la ruta utilizada por el notebook si no forman parte del repositorio.

El notebook está preparado para ejecutarse de arriba hacia abajo.

Los bloques V3 y V4 se mantienen desactivados por defecto mediante:

```python
RUN_EXPERIMENTS = False
```

Esto evita incrementar innecesariamente el tiempo de ejecución de la versión principal del portfolio.

---

# 11. Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- LightGBM
- Matplotlib
- Jupyter Notebook

Modelos y técnicas principales:

- WoE / Information Value
- Regresión logística
- Gradient Boosting
- Modelos de EAD y LGD
- LightGBM
- Regresión cuantílica
- Bootstrap
- Calibración y análisis de probabilidades
- Champion / Challenger
- Expected Credit Loss

---

# 12. Reproducibilidad

El proyecto utiliza una única semilla:

```python
SEED = 1710
```

El split oficial corresponde a:

- **20% Test**
- split aleatorio;
- sin estratificación.

Test no participa en:

- limpieza;
- ingeniería de variables;
- selección;
- entrenamiento;
- calibración del margen.

La pérdida real de Test se reconstruye únicamente cuando el pipeline ya está congelado.

---

# 13. Nota final

La conclusión que más me interesa conservar de este proyecto es sencilla:

> **Un modelo puede estar bien calibrado y aun así resolver mal la decisión que viene después.**

En esta cartera, el mayor salto no vino de utilizar inmediatamente un algoritmo más complejo. Vino de entender mejor **qué estaba optimizando, cuánto costaba cubrir cada riesgo y cómo validar que esa decisión fuera estable**.

Los challengers posteriores muestran que todavía existe margen técnico para mejorar el sistema, pero también confirman que la complejidad adicional solo tiene sentido cuando produce una mejora verificable en la decisión final.
