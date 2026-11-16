# Predicción final – Laboratorio de Implementación I | Analista Junior

Competencia Kaggle: `labo-1-virtual-2026-analista-jr`

## 1. Objetivo

Generar la predicción de bajas (`BAJA+2`) para **septiembre de 2021** (`foto_mes` 202109) en la competencia de Analista Junior,
buscando maximizar la ganancia con un modelo LightGBM optimizado mediante Optimización Bayesiana (BO).

El trabajo se hizo en etapas. En cada experimento se cambió **una sola cosa** respecto del anterior, para poder
atribuir la diferencia de ganancia a ese cambio. Este documento lista todas las corridas, incluidas las primeras
pruebas que se descartaron, y explica cómo se eligió el envío final.

## 2. Punto de partida

- **Punto de partida:** `z739_final_senior.ipynb` de la cátedra (workflow Senior), adaptado al dataset de Analista Junior.
- Todos los cambios en el código están marcados con `>>> CAMBIO <n>` y comentados con el valor original del senior,
  así se pueden encontrar con una búsqueda.
- Los scripts de cada experimento están en `scripts_trabajo/` y se llaman
  `z<experimento>_<descripción>_s<semilla primigenia>.R` (con su `.ipynb` equivalente cuando se corrió como notebook).

### Cambios mínimos para adaptar el senior al junior (presentes en todas las corridas)

| Cambio | Descripción |
|---|---|
| 1 | Semilla primigenia y número de experimento (celda 11) |
| 2 | Dataset `analistajr_competencia_2026.csv.gz` |
| 4 | Competencia `labo-1-virtual-2026-analista-jr` y cortes de envíos del junior |
| 5 | El mensaje de cada envío a Kaggle identifica experimento, semilla y semillerío (`exp=… envios=… semilla=… semBO=… semFT=…`) |
| 6 | Arreglo del `switch` de Data Drifting: `dolar_blue` y `dolar_oficial` llamaban a funciones con otro nombre |

## 3. Metodología general

### 3.1 Datos y particiones

| Etapa | Meses |
|---|---|
| Entrenamiento BO | 201901 – 202105 |
| Validación BO (AUC) | 202107 |
| Entrenamiento final | 201901 – 202107 |
| Predicción (future) | 202109 |
| Clase positiva | `BAJA+1` y `BAJA+2` |

A partir de la 9735, los meses de pandemia 202003 – 202010 se excluyen del entrenamiento de la BO y del final (ver Etapa 2).

### 3.2 Modelo y optimización

- LightGBM, optimizando **AUC** sobre 202107.
- BO con `mlrMBO`: 20 puntos iniciales + **40 iteraciones inteligentes** (senior: 30).
- 5 hiperparámetros optimizados, con los rangos del senior:

| Hiperparámetro | Rango (escala de búsqueda) |
|---|---|
| `num_iterations` | 2^2 – 2^10 |
| `num_leaves` | 2^2 – 2^11 |
| `learning_rate` | 2^-7 – 2^-0.1 |
| `feature_fraction` | 0.05 – 0.90 |
| `min_sum_hessian_in_leaf` | 2^-10 – 2^10 |

- **`training_pct = 1.0`** desde la 9733: la BO usa el 100% de los `CONTINUA` (senior: 0.2).
- Selección de variables con **canaritos** (2 desvíos, ratio 0.2), como en el senior.
- **Semillerío final:** se entrenan N modelos con los mejores hiperparámetros y distintas semillas, y se promedian las probabilidades.
  Las semillas salen de `sample(primos)` con la semilla primigenia, así que dos corridas con la misma primigenia comparten las primeras N semillas.

### 3.3 Envíos a Kaggle

Con la probabilidad promedio se ordena a los clientes y se envían los N con mayor probabilidad.
Al principio se usaron los cortes 1800 – 2400 (de a 100). Desde la 9734 se amplió a **1400 – 2400** (de a 100),
porque el máximo aparecía en el extremo inferior del rango.

## 4. Etapas y corridas

### Etapa 1 – Adaptación del script y primeras pruebas (descartadas)

| Exp. | Script | Qué se probó | Resultado |
|---|---|---|---|
| 9701 | `z_final_junior_semBO.R` | Senior adaptado + **semillerío dentro de la BO** (3 semillas por combinación, se promedia y la BO recibe el AUC del ensemble) + semillerío final 10 | Descartado: no se logró una implementación confiable del semillerío dentro de la BO (ver nota) |
| 9711 | versión preliminar de `z_final_junior_semFT.R` | BO original del senior (un LightGBM por combinación) + semillerío final 30 | Descartado |
| 9721 | `z729_final_junior.ipynb` (cátedra) | **Workflow junior de la cátedra sin cambios**, como referencia | Línea base para comparar; se descartó como camino de trabajo y se continuó con el senior adaptado |
| 9731 | `z_final_junior_semFT.R` | Senior adaptado, BO 40 iteraciones, semillerío final 30, `training_pct` 0.2 | Descartado |
| 9732 | — | Lanzamiento duplicado de la 9731 por error; se abortó | Sin resultados |
| 9733 | `z9733_final_junior_semFT_pct1.R` | Igual a la 9731 pero con `training_pct` **1.0** | Resultado similar a la 9731: el cambio de PCT no movió la ganancia. Se decide mantener 1.0 como base |

**Nota sobre la 9701 (semillerío en la BO).** La idea inicial era que la BO evaluara cada combinación de
hiperparámetros con un ensemble de varias semillas, en lugar de un único LightGBM, para que la búsqueda fuera menos
sensible al azar. Ninguno de los scripts base de la cátedra incluye esta funcionalidad, así que se programó a partir del
senior. La primera versión no dio los resultados esperados y, aunque se intentó corregirla con la consulta al profesor,
no se llegó a una versión validada en el tiempo disponible. Por eso se descartó la idea y se continuó con el script
senior adaptado a junior, con la BO original y el semillerío solo en el entrenamiento final (producción).

Conclusión de la etapa: se trabajó con la BO original y semillerío solo en el entrenamiento final. El senior adaptado sin más cambios quedaba por debajo
de la referencia junior.

### Etapa 2 – Cambios en datos y variables (uno por vez)

| Exp. | Script | Cambio respecto de la anterior | Resultado |
|---|---|---|---|
| 9734 | `z9734_final_junior_semFT_pct1_sinid.R` | 9733 + saco `numero_de_cliente` y `foto_mes` de las variables del modelo (CAMBIO 7, celda 76) | Mejora leve frente a la 9733 |
| 9735 | `z9735_final_junior_semFT_pct1_sinid_pand8.R` | 9734 + **pandemia amplia**: 202003 – 202010 fuera del entrenamiento de BO y final, y los lags que miran esos meses en NA (CAMBIO 8, celdas 51, 74 y 98) | Mejora clara frente a la 9734: el cambio con más impacto de la etapa |
| 9736 | `z9736_final_junior_semFT_pct1_sinid_fe.R` | 9734 + variables históricas `_variacion3`, `_caidas3`, `_pendiente3` y `_ratioavg6` (CAMBIO 9, celda 53), sin pandemia | Similar a la 9734: las variables históricas solas no aportan; se descarta el paquete completo |
| **9737** | `z9737_final_junior_semFT_pct1_sinid_pand8_ratioavg6.R` | 9735 + solo **`_ratioavg6`** = valor actual / promedio de los últimos 6 meses (CAMBIO 9, celda 53) | Mejor resultado hasta ese momento, por encima de todas las anteriores y de la línea base junior (9721). Se adopta como configuración de referencia |
| 9743 | `z9743_like9737_lr9_s523867.R` | 9737 + piso de `learning_rate` en la BO: 2^-9 (senior: 2^-7) (CAMBIO 10, celda 86), porque en la 9737 el `learning_rate` quedó en el límite inferior | El cambio no sumó ganancia (además, menor AUC). Se descarta y se mantiene el rango original |

Conclusión de la etapa: la pandemia amplia mejoró la ganancia (9735 frente a 9734). El paquete completo de variables
históricas sin pandemia no mejoró (9736). Combinando la pandemia con solo `_ratioavg6` se obtuvo el mejor resultado
(9737), que pasó a ser la configuración de referencia para el resto del trabajo. Ampliar el rango de
`learning_rate` sobre la 9737 (9743) no sumó ganancia y se descartó.

> [!IMPORTANT]
> ## Script base para la predicción final
> ## `z9737_final_junior_semFT_pct1_sinid_pand8_ratioavg6.R`

### Etapa 3 – Prueba de ruido: misma configuración, otra semilla primigenia

Para saber cuánto de la ganancia de la 9737 era ruido, se repitió exactamente la misma configuración cambiando
solo la semilla primigenia.

| Exp. | Script | Semilla primigenia | Semillerío final |
|---|---|---|---|
| 9737 | `z9737_final_junior_semFT_pct1_sinid_pand8_ratioavg6.R` | 523867 | 30 |
| 9738 | `z9738_like9737_s237151.R` | 237151 | 30 |
| 9739 | `z9739_like9737_s141971.R` | 141971 | 30 |
| 9740 | `z9740_like9737_s219599.R` | 219599 | 30 |
| 9741 | `z9741_like9737_s981697.R` | 981697 | 30 |
| 9742 | `z9742_like9737_s810013.R` | 810013 | 30 |

Conclusión: con la misma configuración, la ganancia en un mismo corte varía varios puntos según la semilla.
Eso marca el tamaño del ruido: diferencias menores a esa variación entre experimentos no son concluyentes.

### Etapa 4 – Ampliación del semillerío final sobre la 9737

| Exp. | Script | Semilla primigenia | Semillerío final (producción) |
|---|---|---|---|
| 9744 | `z9744_like9737_sem100_s523867.R` | 523867 | 100 |
| 9745 | `z9745_like9737_sem100_s237151.R` | 237151 | 100 |
| 9746 | `z9746_like9737_sem100_s141971.R` | 141971 | 100 |
| 9747 | `z9747_like9737_sem100_s219599.R` | 219599 | 100 |
| 9748 | `z9748_like9737_sem100_s981697.R` | 981697 | 100 |
| 9749 | `z9749_like9737_sem100_s810013.R` | 810013 | 100 |
| 9750 | `z9750_like9737_sem250_s523867.R` | 523867 | 250 |
| 9705 | — | 102191 | 50 |
| 9706 | — | 241109 | 100 |
| 9707 | — | 621749 | 150 |
| 9708 | — | 789047 | 200 |

## 5. Elección del corte

El leaderboard público se calcula sobre un 30% de los clientes, y el privado (resultado final) sobre el 70% restante.
Por eso el valor más alto del público tiene una parte de suerte: elegir el máximo entre muchos envíos es elegir el que
mejor cayó en ese 30%, y esa suerte no se traslada al privado. En vez de elegir el pico, se buscó **el corte que da
ganancia alta de forma estable** en todas las corridas con la configuración de la 9737 (de la 9737 a la 9744, 9747, 9748, 9749,
9705, 9706 y 9707: 14 corridas).

Para cada corte entre 1400 y 2400 se miró, a lo largo de las 14 corridas:
- la **mediana** y el **promedio** de la ganancia;
- el **piso**: el valor que superan 3 de cada 4 corridas (percentil 25);
- **cuántas veces** ese corte quedó entre los 3 mejores de su propia corrida;
- la **dispersión** entre corridas.

Resultados:
- La zona buena es **1600 – 1800**.
- 1600 tiene la mediana más alta y es el corte que más veces queda entre los 3 mejores de su corrida.
- **1800 tiene el piso más alto**: es el corte que menos se cae entre corridas y que casi siempre queda entre los mejores.
- 2100 es el que menos varía, pero porque queda bajo: casi nunca está entre los mejores de su corrida.

## 6. Envío final

Criterios usados, en orden:
1. **Corte 1800**, por ser el más estable en ganancia alta (sección 5).
2. **La corrida con más semillas** en el semillerío final, porque reduce la variación debida al azar.
3. Que su valor público esté cerca del valor típico de ese corte entre todas las corridas, y no sea un pico.

**Envío elegido:** `KA9707_s150_1800.csv` (experimento 9707, semilla primigenia 621749, semillerío final de 150 semillas, corte 1800).
Ganancia en el leaderboard público: **97.279**.
Se adjunta el CSV correspondiente.

## 7. Consideraciones

- Varios experimentos se compararon con la ganancia del leaderboard público, probando distintos cortes. Esto puede
  generar sobreajuste al público; por eso la elección final se basó en la estabilidad entre corridas y no en el máximo.
- La prueba de ruido (Etapa 3) muestra que diferencias pequeñas entre dos corridas no alcanzan para afirmar que una
  configuración es mejor que otra.
- La BO no es exactamente reproducible entre corridas (LightGBM multihilo), aunque se use la misma semilla.
- Si una corrida se interrumpe, el script retoma la BO desde `HT.RDATA` (`mboContinue`) y en el semillerío final saltea
  los modelos ya guardados en `modelos/`. Esto se usó en la 9731, 9736 y 9750 tras cortes de la VM o del kernel.

## 8. Cómo reproducir

1. Copiar el script del experimento (por ejemplo `scripts_trabajo/z9750_like9737_sem250_s523867.ipynb`) a la VM.
2. Verificar en la celda 11 la semilla primigenia y el número de experimento. **Nunca repetir un número de experimento**:
   si la carpeta ya existe, el script retoma lo que haya.
3. Correr todas las celdas (Run All).
4. La salida de cada corrida queda en la VM, en `/content/buckets/b1/exp/WF<experimento>/`, y el script envía cada corte a Kaggle.

### Estructura del repositorio

Repositorio de la competencia final: https://github.com/YVielma/EntregaFinal_lab_YuriVielma_Alexandra_Cremydas

| Carpeta / archivo | Contenido |
|---|---|
| `README.md` | Este documento: descripción del trabajo, corridas y envío final |
| `script_base/` | **Script base de la predicción final:** `z9737_final_junior_semFT_pct1_sinid_pand8_ratioavg6` (`.ipynb` y `.R`) |
| `scripts_trabajo/` | Notebooks (`.ipynb`) y scripts (`.R`) de las corridas 9738 a 9750, hechas a partir del script base |
| `Resultados_Corridas/` | `BO_log.txt` y `PARAM.yml` de cada corrida |
