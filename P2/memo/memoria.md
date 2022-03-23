# Práctica 2: Perceptrón multicapa

Autores: Guillermo García Cobo y Álvaro Zaera de la Fuente

## 1. Algoritmo de retropropagación

a

## 2. Problemas reales

En esta sección, presentamos los resultados obtenidos para los problemas $1$, $2$, $3$ y $5$ para distintas configuraciones de la red.

### Problema real 1

|  LR  | Épocas | Neuronas |                        ECM                        |                        Accuracy                        |
| :--: | :----: | :------: | :-----------------------------------------------: | :----------------------------------------------------: |
| 0.1  |  500   |    2     | ![](problema_real1/ECM_problema_real1_0.1_2.png)  | ![](problema_real1/Accuracy_problema_real1_0.1_2.png)  |
| 0.01 |  500   |    2     | ![](problema_real1/ECM_problema_real1_0.01_2.png) | ![](problema_real1/Accuracy_problema_real1_0.01_2.png) |

A la luz de los resultados, este problema parece el más sencillo de resolver. Con una configuración muy sencilla de tan solo $2$ neuronas logramos un rendimiento muy alto ($>95\%$). Probamos además distintos valores de tasa de aprendizaje (LR), observando que el *overfitting* era menor cuando decrementábamos esta tasa, aunque el rendimiento era ligeramente peor. Elegimos el modelo con menor LR porque creemos que ha sido capaz de generalizar mejor con los datos de validación. A continuación presentamos las matrices de confusión del modelo seleccionado:

|                Matriz de confusión (train)                 |                 Matriz de confusión (val)                 |
| :--------------------------------------------------------: | :-------------------------------------------------------: |
| ![](problema_real1/Matriz_problema_real1_0.01_2_train.png) | ![](problema_real1/Matriz_problema_real1_0.01_2_test.png) |

Tanto en el conjunto de entrenamiento como en el de validación  observamos comportamientos similares, con un gran número de aciertos y los falsos positivos mayores que los falsos negativos en ambos casos, posiblemente porque la clase $0$ está algo sobrerrepresentada.

### Problema real 2



|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real2/ECM_problema_real2_0.01_2_500.png)  | ![](problema_real2/Accuracy_problema_real2_0.01_2_500.png)  |
| 0.01 |  1000  |    2     | ![](problema_real2/ECM_problema_real2_0.01_2_1000.png) | ![](problema_real2/Accuracy_problema_real2_0.01_2_1000.png) |
| 0.01 |  500   |    5     | ![](problema_real2/ECM_problema_real2_0.01_5_500.png)  | ![](problema_real2/Accuracy_problema_real2_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real2/ECM_problema_real2_0.01_10_500.png) | ![](problema_real2/Accuracy_problema_real2_0.01_10_500.png) |
| 0.01 |  500   |    20    | ![](problema_real2/ECM_problema_real2_0.01_20_500.png) | ![](problema_real2/Accuracy_problema_real2_0.01_20_500.png) |

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real2/Matriz_problema_real2_0.01_10_500_train.png) | ![](problema_real2/Matriz_problema_real2_0.01_10_500_test.png) |

### Problema real 3

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real3/ECM_problema_real3_0.01_2_500.png)  | ![](problema_real3/Accuracy_problema_real3_0.01_2_500.png)  |
| 0.01 |  500   |    5     | ![](problema_real3/ECM_problema_real3_0.01_5_500.png)  | ![](problema_real3/Accuracy_problema_real3_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real3/ECM_problema_real3_0.01_10_500.png) | ![](problema_real3/Accuracy_problema_real3_0.01_10_500.png) |

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real3/Matriz_problema_real3_0.01_5_500_train.png) | ![](problema_real3/Matriz_problema_real3_0.01_5_500_test.png) |

### Problema real 5

|  LR  | Épocas | Neuronas |                          ECM                           |                          Accuracy                           |
| :--: | :----: | :------: | :----------------------------------------------------: | :---------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real5/ECM_problema_real5_0.01_2_500.png)  | ![](problema_real5/Accuracy_problema_real5_0.01_2_500.png)  |
| 0.01 |  500   |    5     | ![](problema_real5/ECM_problema_real5_0.01_5_500.png)  | ![](problema_real5/Accuracy_problema_real5_0.01_5_500.png)  |
| 0.01 |  500   |    10    | ![](problema_real5/ECM_problema_real5_0.01_10_500.png) | ![](problema_real5/Accuracy_problema_real5_0.01_10_500.png) |

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
X_{train} =
$$

### Problema real 4 

Comprobamos ahora que la normalización 

|  LR  | Épocas | Neuronas |                            ECM                             |                           Accuracy                           |
| :--: | :----: | :------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
| 0.01 |  500   |    2     | ![](problema_real4/ECM_problema_real4_0.01_2_500_norm.png) | ![](problema_real4/Accuracy_problema_real4_0.01_2_500_norm.png) |
| 0.01 |  500   |    5     | ![](problema_real4/ECM_problema_real4_0.01_5_500_norm.png) | ![](problema_real4/Accuracy_problema_real4_0.01_5_500_norm.png) |

### Problema real 6

En primer lugar, tal y como se nos pide en el enunciado, ejecutamos el problema $6$ con la configuración solicitada. Cabe comentar que el tiempo de ejecución fue de **3 horas y 20 minutos**, demostrando la eficiencia de Julia. Los resultados son los siguientes:

|  LR  | Épocas | Neuronas |                             ECM                             |                           Accuracy                           |
| :--: | :----: | :------: | :---------------------------------------------------------: | :----------------------------------------------------------: |
| 0.1  |  5000  |    20    | ![](problema_real6/ECM_problema_real6_0.1_20_5000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_20_5000_norm.png) |

Además, hemos obtenido las matrices de confusión para esta configuración:

|                 Matriz de confusión (train)                  |                  Matriz de confusión (val)                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](problema_real6/Matriz_problema_real6_0.1_20_5000_norm_train.png) | ![](problema_real6/Matriz_problema_real6_0.1_20_5000_norm_test.png) |



|  LR  | Épocas | Neuronas  |                             ECM                              |                           Accuracy                           |
| :--: | :----: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| 0.1  |  1000  |     2     |  ![](problema_real6/ECM_problema_real6_0.1_2_1000_norm.png)  | ![](problema_real6/Accuracy_problema_real6_0.1_2_1000_norm.png) |
| 0.1  |  1000  |     5     |  ![](problema_real6/ECM_problema_real6_0.1_5_1000_norm.png)  | ![](problema_real6/Accuracy_problema_real6_0.1_5_1000_norm.png) |
| 0.1  |  1000  |    10     | ![](problema_real6/ECM_problema_real6_0.1_10_1000_norm.png)  | ![](problema_real6/Accuracy_problema_real6_0.1_10_1000_norm.png) |
| 0.1  |  1000  | [10, 10]  | ![](problema_real6/ECM_problema_real6_0.1_10-10_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_10-10_1000_norm.png) |
| 0.1  |  1000  | [5, 5, 5] | ![](problema_real6/ECM_problema_real6_0.1_5-5-5_1000_norm.png) | ![](problema_real6/Accuracy_problema_real6_0.1_5-5-5_1000_norm.png) |
