# Geophysical interpreter

## Manuel Ortiz Osio
### e-mail: soff.rpg@gmail.com

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

### Usage

It is a single class that applies a wavelet transform filter and some anomaly detection algorithms. The input is a comma-separated ASCII file with headers and matrix structure `nxm`, where `n` is the number of vectors and `m` is the number of features. For magnetometry the first two columns are expected to be the spatial coordinates of each vector; for *ERT2D* it is expected that the first column is the depth level and the second one its the `x` coordinate of the attribution point, likewise it is expected that the vectors are ordered by level in ascending order. Consult the file [Filtering_example.py](https://github.com/CecilRamza/geophysical_interpreter/blob/main/Filtering example.py "Example of the filtering algorithm").

***

## Preprocessing

### Requirements

- `vtk`
- `random`
- `subprocess`
- `numpy`
- `resipy`
- `scipy`

### Usage

The `preparar` class takes the databases path to train or to cluster: for *magnetometry*, a coma delimited ascii file with a one-row header and three features is expected (two coordinates and a magnetic feature); for *ERT2D* the folder paths in which were the output inversion model files are stored according to the `resipy` API. This class adds the `n` closests neighbors according to the `(x,y)` coordinates and applies mathematical transformations as well.

The `modelar_electrica_2D` class takes the mesh first and last elements with a `VTK` format created with the `resipy` API, as well the resistivity scalar data. The direct models are made with random resistivity values chosen between a defined range, by means of the automatic sequence files according to the `resipy` API, as well with the survey geometry.

Consult the file [Preprocessing_example.py](https://github.com/CecilRamza/geophysical_interpreter/blob/main/Preprocessing_example.py "Example of the preprocessing algorithm").

***

## Training

### Requirements

- `random`
- `numpy`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`
- `mpl_toolkits`
- `tools.py`

### Usage

Takes the comma separated ascii files without headers, and a list with the features -column number- that will be used to train. This class contains the following methods:

- to reduce the dimensionality with the *PCA* algorithm,
- variable scaling,
- data normalization,
- unsupervized training with the *k-means*, *k-medians* and *SOM*

Consult the file [Training_example.py](https://github.com/CecilRamza/geophysical_interpreter/blob/main/Training_example.py "Example of the training algorithm").

***

## Clustering

### Requeriments

- `vtk`
- `math`
- `colorsys`
- `numpy`
- `pyvista`
- `minisom`
- `matplotlib`
- `sklearn`
- `pyclustering`

### Usage

This class performs the clustering on the input database, applying the respective scaling and dimensionality reduction used on the training set. Consult the file [Clustering_example.py](https://github.com/CecilRamza/geophysical_interpreter/blob/main/Clustering_example.py "Example of the clustering methodology") using `pyvista` to display the output images.
