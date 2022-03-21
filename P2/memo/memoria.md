# Práctica 2: Perceptrón multicapa

Autores: Guillermo García Cobo y Álvaro Zaera de la Fuente

## 1. Algoritmo de retropropagación

a

## 2. Problemas reales

En esta sección, presentamos los resultados obtenidos para los problemas $1$, $2$, $3$ y $5$ para distintas configuraciones de la red.

### Problema real 1

|  LR  | Épocas | Neuronas | ECM  | Accuracy | Matriz de confusión |
| :--: | :----: | :------: | :--: | :------: | :-----------------: |
| 0.1  |  500   |    2     |      |          |                     |
| 0.01 |  500   |    2     |      |          |                     |

A la luz de los resultados, este problema parece el más sencillo de resolver. Con una configuración muy sencilla logramos un rendimiento muy alto

### Problema real 2

|  LR  | Épocas | Neuronas | ECM  | Accuracy | Matriz de confusión |
| :--: | :----: | :------: | :--: | :------: | :-----------------: |
| 0.1  |  500   |    2     |      |          |                     |
| 0.01 |  500   |    2     |      |          |                     |

### Problema real 3

|  LR  | Épocas | Neuronas | ECM  | Accuracy | Matriz de confusión |
| :--: | :----: | :------: | :--: | :------: | :-----------------: |
| 0.1  |  500   |    2     |      |          |                     |
| 0.01 |  500   |    2     |      |          |                     |

### Problema real 5

|  LR  | Épocas | Neuronas | ECM  | Accuracy | Matriz de confusión |
| :--: | :----: | :------: | :--: | :------: | :-----------------: |
| 0.1  |  500   |    2     |      |          |                     |
| 0.01 |  500   |    2     |      |          |                     |



## 3. Problemas reales y normalización

Como se indica en el enunciado, para los problemas $4$ y $6$ no es posible encontrar una configuración que dé buenos resultados. A continuación mostramos los resultados de rendimiento para una de las configuraciones que hemos probado:

| Problema 4 | Problema 6 |
| :--------: | :--------: |
|            |            |

Para determinar la causa de que la red no sea capaz de aprender, hemos observado la distribución de las variables de entrada. Creemos que el *boxplot* es una buena forma de agrupar la información que queremos observar y de poder ver diferencias entre las distintas variables. Además, incluimos la gráfica para otros problemas sí resolubles (el $1$ y el $2$) para comprobar porqué el $4$ y el $6$ no lo son.

|           Problema 1           |           Problema 2           |           Problema 4           |           Problema 6           |
| :----------------------------: | :----------------------------: | :----------------------------: | :----------------------------: |
| ![](img/problema1_boxplot.png) | ![](img/problema2_boxplot.png) | ![](img/problema4_boxplot.png) | ![](img/problema6_boxplot.png) |

En la figura anterior podemos ver con claridad la razón que causa que los primeros dos problemas sean resolubles pero los dos siguientes no. Las variables del problema $1$ y el $2$ están definidas en el intervalo $[0,1]$, con medias y varianzas parejas. Sin embargo, en el problema $4$, el intervalo de definición de la primera variable es mucho mayor que el de las demás, además de contar con una varianza muy superior. Esta variable claramente dificultará el aprendizaje, ya que los pesos deberán adaptarse a esta entrada tan distinta del resto, a la que al principio se le dará demasiada importancia al tener más magnitud. En cuanto al problema $6$, cada variable presenta un intervalo de definición y varianza distintas. El problema más importante es el hecho de la magnitud de todas las variables, muy superior a las de los problemas $1$ y $2$. Esto causará que los pesos deban ser reducidos enormemente para adaptarse, lo que dificultará la convergencia.

|  LR  | Épocas | Neuronas | ECM  | Accuracy | Matriz de confusión |
| :--: | :----: | :------: | :--: | :------: | :-----------------: |
| 0.1  |  500   |    2     |      |          |                     |
| 0.01 |  500   |    2     |      |          |                     |

