# Introducción a las Redes Neuronales Artificiales

Autores: Guillermo García Cobo y Álvaro Zaera de la Fuente

## 1. Implementación de una librería para el manejo de redes neuronales

Con el objetivo de sacar el máximo provecho a estas prácticas, hemos decidido desarrollar el código de las mismas en [Julia](https://julialang.org/). Con esto, además de aprender un lenguaje nuevo, conseguimos una implementación muy eficiente de la librería que se nos requiere. Si no tiene instalado Julia, puede obtenerlo en el [siguiente enlace](https://julialang.org/downloads/).

La implementación de esta librería puede encontrarse en los paquetes `RedNeuronal_pkg.jl`, `Capa_pkg.jl`, `Neurona_pkg.jl`.

## 2. Neuronas de McCulloch-Pitts

![](Red.png)

Usando la librería desarrollada en el apartado anterior, se ha implementado en `FrioCalor.jl` la red de McCulloch-Pitts que se ve en la imagen. Por tanto, las conexiones y los pesos usados en nuestro diseño son los que se incluyen en la imagen, mientras que el valor que hemos asignado al sesgo ($\theta$) es $2$.

El objetivo de esta red es producir la siguiente salida:

* $Y_1(t)$ se activa si se detecta calor en $X_1(t-1)$ o si se detecta frío en $X_2(t-3)$ pero no en $X_2(t-2)$.
* $Y_2(t)$ se activa si se detecta frío en $X_2(t-1)$ y en $X_2(t-2)$.

En términos de puertas lógicas, debemos construir lo siguiente:

* $Y_1(t) = X_1(t-1) \or (X_2(t-3) \and \neg X_2(t-2))$ 
* $Y_2(t) = X_2(t-1) \and X_2(t-2)$

Con esto en mente, el diseño es válido porque implementa las puertas lógicas que equivalen al funcionamiento buscado. Para que esto sea así, es importante fijar el valor del sesgo a $2$, dado que sino las neuronas no se activarían cuando deben. Por ejemplo, $Y_2$ solo se activa si $X_2(t-1) + X_2(t-2) \ge \theta$ , y como las entradas son binarias, es necesario que $\theta = 2$ para que la salida sea el AND buscado. El razonamiento para el OR es equivalente, solo que en este caso los pesos valen $2$ para que la salida se active cuando al menos una de las entradas esté activada. La negación la implementamos restando la entrada, ya que si esta es $0$, la resta no afecta y la neurona se activa. Concluimos entonces que el diseño es válido. Para convencernos de ello, exponemos a continuación la salida con las entradas que se proponen en el enunciado.

| $x_1$ | $x_2$ | $z_1$ | $z_2$ | $y_1$ | $y_2$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
|   0   |   0   |   0   |   0   |   0   |   0   |
|   1   |   1   |   0   |   0   |   0   |   0   |
|   1   |   1   |   0   |   1   |   1   |   0   |
|   0   |   1   |   0   |   1   |   1   |   1   |
|   1   |   0   |   0   |   1   |   0   |   1   |
|   0   |   0   |   1   |   0   |   1   |   0   |
|   0   |   0   |   0   |   0   |   1   |   0   |
|   0   |   0   |   0   |   0   |   0   |   0   |
|   0   |   0   |   0   |   0   |   0   |   0   |
|   0   |   0   |   0   |   0   |   0   |   0   |

Obsérvese que la salida sigue las reglas deseadas. Esto es: 

* $Y_1(2) = 1$, ya que $X_1(1) = 1$
* $Y_2(3) = 1$, ya que $X_2(2)=1$ y $X_2(1)=1$
* $Y_1(6)=1$, ya que $X_2(4)=0$ y $X_2(3)=1$

Como hemos visto, dada una entrada, la salida de la red será la siguiente. Si la entrada muestra calor, entonces la salida de calor $Y_1$ se activará. En cuanto a la dependencia de las entradas de frío, la salida dependerá de los estados anteriores siguiendo las reglas que hemos descrito al principio de este apartado.

## 3. Lectura de datos

La funcionalidad requerida en este apartado está implementada en `LecturaDatos.jl`.

## 4. Perceptrón y Adaline

### 4.1 Problemas lógicos

A continuación mostramos las fronteras de decisión obtenidas con el Perceptrón y el Adaline para cada uno de los tres problemas lógicos resolubles linealmente:

|               Perceptrón               |               Adaline               |
| :------------------------------------: | :---------------------------------: |
| ![](Perceptrón AND_decision_line.png)  | ![](Adaline AND_decision_line.png)  |
|  ![](Perceptrón OR_decision_line.png)  |  ![](Adaline OR_decision_line.png)  |
| ![](Perceptrón NAND_decision_line.png) | ![](Adaline NAND_decision_line.png) |

Como el lector habrá podido observar, hay un problema lógico que no está presente en las imágenes anteriores. Este es el XOR, que no es resoluble linealmente. Como se ve en la siguiente imagen, no es posible trazar una recta que separe las dos clases presentes.

![](XOR_decision_line.png)

Para solucionar este problema, debemos usar técnicas que sean capaces de separar de formas no lineales las distintas clases. Una alternativa para esto es implementar una red neuronal profunda con al menos una capa oculta con activación no lineal.

### 4.2 Problema real 1



### 4.3 Problema real 2



