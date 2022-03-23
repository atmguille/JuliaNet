# Práctica 2: Perceptrón multicapa

Autores: Guillermo García Cobo y Álvaro Zaera de la Fuente

## 1. Algoritmo de retropropagación

a

## 2. Problemas reales

En esta sección, presentamos los resultados obtenidos para los problemas $1$, $2$, $3$ y $5$ para distintas configuraciones de la red. Todas las pruebas han sido ejecutadas en modo $1$ para contar con un conjunto de validación, que en todos los casos ha sido del $70\%$.

### Problema real 1

|  LR  | Épocas | Neuronas |                        ECM                        |                        Accuracy                        |
| :--: | :----: | :------: | :-----------------------------------------------: | :----------------------------------------------------: |
| 0.1  |  500   |    2     | ![](problema_real1/ECM_problema_real1_0.1_2.png)  | ![](problema_real1/Accuracy_problema_real1_0.1_2.png)  |
| 0.01 |  500   |    2     | ![](problema_real1/ECM_problema_real1_0.01_2.png) | ![](problema_real1/Accuracy_problema_real1_0.01_2.png) |

A la luz de los resultados, este problema parece el más sencillo de resolver. Con una configuración muy sencilla de tan solo $2$ neuronas logramos un rendimiento muy alto ($>95\%$). Probamos además distintos valores de tasa de aprendizaje (LR), observando que el *overfitting* era menor cuando decrementábamos esta tasa, aunque el rendimiento era ligeramente peor. Elegimos el modelo con menor LR porque creemos que ha sido capaz de generalizar mejor con los datos de validación. A continuación presentamos las matrices de confusión del modelo seleccionado:

|                Matriz de confusión (train)                 |                 Matriz de confusión (val)                 |
| :--------------------------------------------------------: | :-------------------------------------------------------: |
| ![](problema_real1/Matriz_problema_real1_0.01_2_train.png) | ![](problema_real1/Matriz_problema_real1_0.01_2_test.png) |

Tanto en el conjunto de entrenamiento como en el de validación observamos comportamientos similares, con un gran número de aciertos y los falsos positivos mayores que los falsos negativos en ambos casos.

### Problema real 2

Como en la sección anterior hemos concluido que el mejor LR era $0.01$, las pruebas a continuación se harán fijando este valor.

En primer lugar, probamos a aumentar el número de épocas, concluyendo que, si bien mejora ligeramente aumentando el número de épocas, la mejora nos es sustancial. Por ello, las pruebas a continuación se harán con $500$ épocas.

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real2/ECM_problema_real2_0.01_2_500.png)  | ![](problema_real2/Accuracy_problema_real2_0.01_2_500.png)  |
| 0.01 |  1000  |    2     | ![](problema_real2/ECM_problema_real2_0.01_2_1000.png) | ![](problema_real2/Accuracy_problema_real2_0.01_2_1000.png) |

Como vemos en la siguiente tabla, según aumentamos el número de neuronas el ECM final de entrenamiento disminuye. Sin embargo, aumentar en exceso el número de neuronas también empeora la capacidad de generalización, tal y como se puede observar en la ejecución con $20$ neuronas. Concluimos que el mejor modelo es el de $10$ neuronas, un número lo suficientemente alto para aprender correctamente, pero lo suficientemente pequeño para no sobre-aprender.

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    5     | ![](problema_real2/ECM_problema_real2_0.01_5_500.png)  | ![](problema_real2/Accuracy_problema_real2_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real2/ECM_problema_real2_0.01_10_500.png) | ![](problema_real2/Accuracy_problema_real2_0.01_10_500.png) |
| 0.01 |  500   |    20    | ![](problema_real2/ECM_problema_real2_0.01_20_500.png) | ![](problema_real2/Accuracy_problema_real2_0.01_20_500.png) |

A continuación, presentamos la matriz de confusión para el modelo seleccionado ($10$ neuronas). Es evidente que tenemos más falsos positivos que falsos negativos, posiblemente debido a que la clase $1$ es claramente mayoritaria. Por esta razón, el modelo tenderá a predecir en mayor medida la clase $1$, y quizás no aprenda correctamente las características intrínsecas de la clase $0$.

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real2/Matriz_problema_real2_0.01_10_500_train.png) | ![](problema_real2/Matriz_problema_real2_0.01_10_500_test.png) |

### Problema real 3

En la siguiente tabla observamos los resultados para distintas configuraciones sobre los datos del problema $3$. Una vez más, el número intermedio de neuronas ($5$) es el que nos da mejores resultados en cuánto a error y generalización.

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real3/ECM_problema_real3_0.01_2_500.png)  | ![](problema_real3/Accuracy_problema_real3_0.01_2_500.png)  |
| 0.01 |  500   |    5     | ![](problema_real3/ECM_problema_real3_0.01_5_500.png)  | ![](problema_real3/Accuracy_problema_real3_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real3/ECM_problema_real3_0.01_10_500.png) | ![](problema_real3/Accuracy_problema_real3_0.01_10_500.png) |

Una vez más, presentamos la matriz de confusión del modelo elegido. En este caso, la matriz es $3 \times 3$, ya que los datos cuentan con $3$ clases a predecir. Observamos que la clasificación es casi perfecta, y únicamente presenta un número bajo de fallos en la clase $1$, para la que el modelo predice la clase $2$.

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real3/Matriz_problema_real3_0.01_5_500_train.png) | ![](problema_real3/Matriz_problema_real3_0.01_5_500_test.png) |

### Problema real 5

En este caso, presentamos los resultados para $2$, $5$, $10$ y $20$ neuronas. En este caso, los resultados entre $5$ y $10$ neuronas son muy parejos, aunque parecen muy ligeramente superiores para $10$ neuronas. Además, en $20$ neuronas es muy claro el *overfitting*. Por esto, como $10$ es el último mejor resultado y su rendimiento es algo superior, los consideramos como el mejor modelo.

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real5/ECM_problema_real5_0.01_2_500.png)  | ![](problema_real5/Accuracy_problema_real5_0.01_2_500.png)  |
| 0.01 |  500   |    5     | ![](problema_real5/ECM_problema_real5_0.01_5_500.png)  | ![](problema_real5/Accuracy_problema_real5_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real5/ECM_problema_real5_0.01_10_500.png) | ![](problema_real5/Accuracy_problema_real5_0.01_10_500.png) |
| 0.01 |  500   |    20    | ![](problema_real5/ECM_problema_real5_0.01_20_500.png) | ![](problema_real5/Accuracy_problema_real5_0.01_20_500.png) |

La matriz de confusión para el modelo de $10$ neuronas se presenta a continuación. Observamos como una vez más los falsos positivos superan a los falsos negativos.

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real5/Matriz_problema_real5_0.01_10_500_train.png) | ![](problema_real5/Matriz_problema_real5_0.01_10_500_test.png) |

## 3. Problemas reales y normalización

Como se indica en el enunciado, para los problemas $4$ y $6$ no es posible encontrar una configuración que dé buenos resultados. A continuación mostramos los resultados de rendimiento para una de las configuraciones que hemos probado:

|                      Problema 4                       |                      Problema 6                       |
| :---------------------------------------------------: | :---------------------------------------------------: |
| ![](problema_real4/ECM_problema_real4_0.01_2_500.png) | ![](problema_real6/ECM_problema_real6_0.1_2_1000.png) |

Para determinar la causa de que la red no sea capaz de aprender, hemos observado la distribución de las variables de entrada. Creemos que el *boxplot* es una buena forma de agrupar la información que queremos observar y de poder ver diferencias entre las distintas variables. Además, incluimos la gráfica para otros problemas sí resolubles (el $1$ y el $2$) para comprobar porqué el $4$ y el $6$ no lo son.

|           Problema 1           |           Problema 2           |           Problema 4           |           Problema 6           |
| :----------------------------: | :----------------------------: | :----------------------------: | :----------------------------: |
| ![](img/problema1_boxplot.png) | ![](img/problema2_boxplot.png) | ![](img/problema4_boxplot.png) | ![](img/problema6_boxplot.png) |

En la figura anterior podemos ver con claridad la razón que causa que los primeros dos problemas sean resolubles pero los dos siguientes no. Las variables del problema $1$ y el $2$ están definidas en el intervalo $[0,1]$, con medias y varianzas parejas. Sin embargo, en el problema $4$, el intervalo de definición de la primera variable es mucho mayor que el de las demás, además de contar con una varianza muy superior. Esta variable claramente dificultará el aprendizaje, ya que los pesos deberán adaptarse a esta entrada tan distinta del resto, a la que al principio se le dará demasiada importancia al tener más magnitud. En cuanto al problema $6$, cada variable presenta un intervalo de definición y varianza distintas. El problema más importante es el hecho de la magnitud de todas las variables, muy superior a las de los problemas $1$ y $2$. Esto causará que los pesos deban ser reducidos enormemente para adaptarse, lo que dificultará la convergencia.

Como solución, procedemos a normalizar los valores de entrada como sigue:
$$
X_{train} = \frac{X_{train} - \mu_{train}}{\sigma_{train}}\\
X_{test} = \frac{X_{test} - \mu_{train}}{\sigma_{train}},
$$

es decir, las entradas del conjunto de test son normalizadas con la media y la desviación del conjunto de entrenamiento.

### Problema real 4 

Comprobamos ahora que la normalización es efectiva en el problema $4$. En la siguiente tabla observamos los resultados para $2$ y $5$ neuronas, muy superiores a los que hemos visto unas líneas más arriba.

|  LR  | Épocas | Neuronas |                            ECM                             |                           Accuracy                           |
| :--: | :----: | :------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real4/ECM_problema_real4_0.01_2_500_norm.png) | ![](problema_real4/Accuracy_problema_real4_0.01_2_500_norm.png) |
| 0.01 |  500   |    5     | ![](problema_real4/ECM_problema_real4_0.01_5_500_norm.png) | ![](problema_real4/Accuracy_problema_real4_0.01_5_500_norm.png) |

### Problema real 6

En primer lugar, tal y como se nos pide en el enunciado, ejecutamos el problema $6$ con la configuración solicitada. Cabe comentar que el tiempo de ejecución fue de **3 horas y 20 minutos TODO!!!**, demostrando la eficiencia de Julia. Los resultados son los siguientes:

|  LR  | Épocas | Neuronas |                             ECM                             |                           Accuracy                           |
| :--: | :----: | :------: | :---------------------------------------------------------: | :----------------------------------------------------------: |
| 0.1  |  5000  |    20    | ![](problema_real6/ECM_problema_real6_0.1_20_5000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_20_5000_norm.png) |

Observamos cómo, al igual que en el problema $4$, el modelo es capaz de aprender tras normalizar los datos de entradas. Además, hemos obtenido las matrices de confusión para esta configuración:

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real6/Matriz_problema_real6_0.1_20_5000_norm_train.png) | ![](problema_real6/Matriz_problema_real6_0.1_20_5000_norm_test.png) |

Una vez más, la tendencia de errores es la misma en el conjunto de entrenamiento que en el conjunto de validación. Parece que en este caso, la clase $0$ está sobre-representada, de ahí que haya más falsos negativos.

A continuación, presentamos los resultados para distintas variaciones en el número de neuronas de la configuración que se nos solicitaba con $1000$ épocas. Observamos cómo el rendimiento va creciendo según aumentamos el número de neuronas. ... **TODO: AÑADIR COMENTARIOS DE 30 NEURONAS**.

|  LR  | Épocas | Neuronas |                             ECM                             |                           Accuracy                           |
| :--: | :----: | :------: | :---------------------------------------------------------: | :----------------------------------------------------------: |
| 0.1  |  1000  |    2     | ![](problema_real6/ECM_problema_real6_0.1_2_1000_norm.png)  | ![](problema_real6/Accuracy_problema_real6_0.1_2_1000_norm.png) |
| 0.1  |  1000  |    5     | ![](problema_real6/ECM_problema_real6_0.1_5_1000_norm.png)  | ![](problema_real6/Accuracy_problema_real6_0.1_5_1000_norm.png) |
| 0.1  |  1000  |    10    | ![](problema_real6/ECM_problema_real6_0.1_10_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_10_1000_norm.png) |
| 0.1  |  1000  |    20    | ![](problema_real6/ECM_problema_real6_0.1_20_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_20_1000_norm.png) |
| 0.1  |  1000  |    30    | ![](problema_real6/ECM_problema_real6_0.1_30_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_30_1000_norm.png) |

Como es evidente que este es el problema más desafiante de todos, hemos decidido probar la funcionalidad de varias capas sobre este conjunto de datos. Los resultados se muestran en la siguiente tabla:

|  LR  | Épocas |  Neuronas  |                             ECM                              |                           Accuracy                           |
| :--: | :----: | :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 0.1  |  1000  |  [10, 10]  | ![](problema_real6/ECM_problema_real6_0.1_10-10_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_10-10_1000_norm.png) |
| 0.1  |  1000  | [5, 5, 5]  | ![](problema_real6/ECM_problema_real6_0.1_5-5-5_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_5-5-5_1000_norm.png) |
| 0.1  |  1000  | [5, 10, 5] | ![](problema_real6/ECM_problema_real6_0.1_5-10-5_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_5-10-5_1000_norm.png) |

Con dos capas ocultas de $10$ neuronas cada una, el rendimiento es similar al de una sola capa. Sin embargo, si añadimos una tercera capa oculta, la complejidad del modelo es tal que parece impedir que se aproveche el mayor número de parámetros para un mejor aprendizaje.

Por último, como en las pruebas realizadas con los otros problemas el LR fijado a $0.01$ parecía tener un rendimiento superior, hemos decidido hacer probar a modificar este parámetro en la configuración solicitada.

|  LR  | Épocas | Neuronas |                             ECM                              |                           Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 0.01 |  1000  |    20    | ![](problema_real6/ECM_problema_real6_0.01_20_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.01_20_1000_norm.png) |

Para $1000$ épocas el rendimiento es muy superior al de la configuración solicitada, por lo que hemos decidido ejecutarlo para $5000$ épocas. Estos resultados, una vez más muy superiores, se muestran a contunuación:

|  LR  | Épocas | Neuronas |                             ECM                              |                           Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 0.01 |  5000  |    20    | ![](problema_real6/ECM_problema_real6_0.01_20_5000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.01_20_5000_norm.png) |

**TODO: añadir más comentarios si procede. **