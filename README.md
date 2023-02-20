# Geophysical interpreter

## Manuel Ortiz Osio
## e-mail: soff.rpg@gmail.com

Content:

- **`filtering.py`:** code implemented to perform the filtering process.
- **`preprocessing.py`:** prepare the input databases for further processing.
- **`training.py`:** code implemented to perform the unsupervised training algorithm
- **`clustering.py`:** applies the SOM or the clusters to perform the clustering.
- ***`tools.py:`*** contains the functions of general purpose.

as well as some simple examples for each processing phase.

***

## Filtering

### Requirements

- `pywt`
- `math`
- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`
- `tools.py`

### Use

It is a single class that applies a wavelet transform filter and some anomaly detection algorithms. The input is a comma-separated ASCII file with headers and matrix structure `nxm`, where `n` is the number of vectors and `m` is the number of features. For magnetometry the first two columns are expected to be the spatial coordinates of each vector; for *ERT2D* it is expected that the first column is the depth level and the second one its the `x` coordinate of the attribution point, likewise it is expected that the vectors are ordered by level in ascending order. Consult the file [Ejemplo_filtrado.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_filtrado.py "Ejemplos del filtrado implementado").

***

## Preproceso

### Requerimientos

- `vtk`
- `random`
- `subprocess`
- `numpy`
- `resipy`
- `scipy`

### Uso

La clase `preparar` recibe como argumento la ruta a las bases de datos que se usarán para entrenar o agrupar. Para magnetometría se reciben los nombres de los archivos que se quieran preparar, se considera un archivo ascii separado por comas con un encabezado y 3 columnas; mientras que para *TRE2D* se reciben las carpetas en donde se encuentran los archivos de salida de la inversión de datos, considerando que este proceso se realizó con la biblioteca `resipy`. Esta clase se encarga de encontrar los vecinos espacialmente más cercanos y de aplciar transformaciones.

La clase `modelar_electrica_2D` recibe los elementos inicial y final de las mallas vtk creadas con la biblioteca `resipy`, así como la malla en su totalidad incluyendo los valores de resistividad. Posteriormente se modela usando valores de resistividad aleatorios dentro de un intervalo definido usando secuencias de lectura con la estructura requerida por la biblioteca `resipy`, así mismo se requiere de la geometría del levantamiento. Finalmente se agrupan los modelos realizados en un solo archivo.

Consultar el archivo [Ejemplo_preparacion.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_preparacion.py "Ejemplos para preparar las bases de datos").

***




## Entrenamiento

### Requerimientos

- `random`
- `numpy`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`
- `mpl_toolkits`
- `herramientas.py`

### Uso

Se trata de una sola clase, se reciben los archivos ascii separados por comas sin encabezados, y una lista con las columnas con las que se desea entrenar. Esta clase contiene los métodos para reducir la dimensionalidad mediante *PCA*, escalamiento y normalización de los datos, el entrenamiento mediante métodos de agrupamiento y el entrenamiento mediante *SOM*. Consultar el archivo [Ejemplo_entrenamiento.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_entrenamiento.py "Ejemplos para aplicar entrenamiento").

***

## Agrupamiento

### Requerimientos

- `vtk`
- `math`
- `colorsys`
- `numpy`
- `pyvista`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`

### Uso

Esta clase realiza el agrupamiento a la base de datos ingresada, aplicándole el respectivo escalamiento y reducciones usadas en el conjunto de entrenamiento. Consultar el archivo [Ejemplo_agrupamiento.py](https://github.com/CecilRamza/Interprete_geofisica/blob/main/Ejemplo_agrupamiento.py "Ejemplos de despliegue de resultados de agrupamiento") para ejemplos en el despliegue de resultados usando `pyvista`.
