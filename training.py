import random
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from matplotlib import cm, colorbar
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from matplotlib.patches import RegularPolygon
from pyclustering.cluster.kmedians import kmedians
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tools import *

########################################################
#                                                      #
# This class prepares the database to train with the   #
# k-means, k-medians and SOM.                          #
#                                                      #
# entrenamiento(archivos,columnas), where "archivos"   #
# is a list that contains the file path of the         #
# databases to train, "columnas" is a list with the    #
# column number with the variables to train the system.#
#                                                      #
# entrenamiento.reduccion_pca(columnas,porcentajes),   #
# where "columnas" is a list that contains the column  #
# number of the variables that the user wants to       #
# reduce the dimensionality, "porcentajes" is a        #
# list with the minimum variance's percentages to      #
# calculate the number of princilal components.        #
# entrenamiento.pcas is a list with the principal      #
# input database, its purpose is to use the same PCA   #
# algorithm output to apply on the clustering set.     #
# entrenamiento.estandarizaciones is a list with the   #
# standarization parameters for each input variable    #
# set, its purpose is to apply this parameters on the  #
# clustering set. entrenar.componentes is the number   #
# of components of each input variable set, its purpose#
# is to apply on the input variable set.               #
#                                                      #
# entrenamiento.escalamiento(metodo), where "metodo"   #
# is the method to scale the values of each variable,  #
# the options are: "estandarizar", "minmax" and        #
# "maxabs". entrenamiento.escala stores the model to be#
# use on the clustering set.                           #
#                                                      #
# entrenamiento.kmeans(num_cum), where "num_cum" is the#
# number of clusters to train the system.              #
# entrenamiento.kmedias stores the clusters to apply on#
# the clustering set.                                  #
#                                                      #
# entrenamiento.kmedians(num_cum), where "num_cum" is  #
# the number of clusters to train the system.          #
# entrenamiento.kmedianas stores the clusters to apply #
# on the clustering set.                               #
#                                                      #
# entrenamiento.som(x,y,iteraciones,vecindad,aprendizaje,topologia), #
# where "x" and "y" are the map dimensions, "vecindad" #
# is the neighborhood parameter to update the neurons  #
# weights, "aprendizaje" is the learning factor        #
# "topologia" is the network topology, that could be   #
# "hexagonal" or "rectangular". entrenamiento.mau      #
# stores the trained map. The u-matrix and the error   #
# plots are generated.                                 #
#                                                      #
########################################################

class entrenamiento:
    def __init__(self,archivos,columnas,escalamiento_min=[],escalamiento_max=[]):
        tamanos=[]
        for ijk in range(len(archivos)): #Ciclo que itera sobre el n??mero de archivos
            d=open(archivos[ijk])
            datos=np.genfromtxt(d,delimiter=",") #Carga del archivo correspondiente
            if(ijk==0): #Si es el primer archivo, se inicializa el arreglo - tama??o de las variables
                n=datos.shape[1]
                self.conj_entr=np.zeros((0,n))
            tamanos.append(datos.shape[0])
            self.conj_entr=np.append(self.conj_entr,datos,axis=0) #Se almacena todo el contenido del archivo
            d.close()
        mascara=np.zeros(n,dtype=bool) #Inicializaci??n de mascara booleana de tama??o igual al n??mero de variables
        mascara[columnas]=True #Las columnas elegidas se marcan como verdaderas
        self.conj_entr=self.conj_entr[:,mascara] #Se filtran las variables elegidas
        self.nn=self.conj_entr.shape[0] #Se almacena el tama??o de la base de datos
        n=self.conj_entr.shape[1]
        if(len(escalamiento_min)>0 and len(escalamiento_max)>0 and len(escalamiento_min)==len(escalamiento_max)):
            inferior=0
            for k in range(len(tamanos)):
                superior=inferior+tamanos[k]
                for i in range(n):
                    pivote=self.conj_entr[inferior:superior,i]
                    rmin=min(pivote)
                    rmax=max(pivote)
                    for j in range(pivote.shape[0]):
                        pivote[j]=(pivote[j]-rmin)/(rmax-rmin)*(escalamiento_max[i]-escalamiento_min[i])+escalamiento_min[i]
                    self.conj_entr[inferior:superior,i]=pivote
                inferior=superior
    def reduccion_pca(self,columnas,porcentajes):
        sin_cambios=np.arange(self.conj_entr.shape[1]) #Arreglo de tama??o igual al n??mero de columnas
        npo=len(porcentajes) #N??mero de reducciones deseadas
        for columna in columnas: #Ciclo que itera sobre el conjunto de columnas deseadas
            for digito in columna: #Ciclo que itera sobre cada columna del conjunto de columnas
                if digito in sin_cambios: #Si la columna se encuentra en el arreglo que contiene todas las columnas, se elimina dicha columna
                    sin_cambios=np.delete(sin_cambios,np.where(sin_cambios==digito)[0])
        conjunto=self.conj_entr[:,sin_cambios] #Se almacena la base de datos que no ser?? reducida
        self.pcas=[] #Inicializaci??n de una lista que contendr?? las componentes principales de cada conjunto deseado
        self.estandarizaciones=[] #Inicializaci??n de una lista que contendr?? la estandarizaci??n aplicada a cada conjunto
        for columna in columnas: #Ciclo que itera sobre los conjuntos de columnas
            arreglo=self.conj_entr[:,columna] #Arreglo auxiliar que almacena cada conjunto a aplicar pca
            escala=preprocessing.StandardScaler().fit(arreglo) #Estandarizaci??n
            self.estandarizaciones.append(escala) #Se a??ade la estandarizaci??n anterior al arreglo que las almacena
            arreglo=escala.transform(arreglo) #Se aplica la estandarizaci??n
            pca=PCA() #Inicializaci??n de pca
            self.pcas.append(pca.fit(arreglo)) #Se a??aden los par??metros del pca actual al arreglo que los almacena
            self.componentes=np.zeros(npo) #Inicializaci??n de arreglo para almacenar el n??mero de componentes reducidas para cada conjunto ingresado
        for i in range(npo): #Ciclo que itera sobre el n??mero de conjuntos solicitados
            varianzas=self.pcas[i].explained_variance_ratio_ #Arreglo que almacena las varianzas de cada componente para el conjunto actual
            suma=0 #Variable auxiliar para sumatoria de varianza
            for varianza in varianzas: #Ciclo que itera sobre la varianza de cada componente calculada
                self.componentes[i]=self.componentes[i]+1 #Se mide el n??mero de componentes
                suma=suma+varianza #Sumatoria de la varianza
                if suma>=porcentajes[i]: #Si la suma de las varianzas es igual o mayor al umbral deseado se sale del ciclo
                    break
            pivote=self.conj_entr[:,columnas[i]]
            arreglo=np.matmul(pivote,np.transpose(self.pcas[i].components_[0:int(self.componentes[i]),:])) #Se aplica la reducci??n de pca considerando las componentes obtenidas del ciclo anterior
            arr=arreglo.reshape((self.nn,int(self.componentes[i]))) #Se cambian las dimensiones del arreglo anterior para concordar con la base de datos
            conjunto=np.append(conjunto,arr,axis=1) #Al conjunto que no recibe cambios se le agregan las columnas reducidas
        self.conj_entr=conjunto #Al final, el conjunto de entrenamiento se iguala al conjunto procesado
    def escalamiento(self,metodo="maxabs"):
        if(metodo=="estandarizar"): #Estandarizaci??n
            self.escala=preprocessing.StandardScaler().fit(self.conj_entr) #Escala
            self.conj_entr=self.escala.transform(self.conj_entr) #Aplicaci??n
        elif(metodo=="minmax"): #Escalado entre -1 y 1
            self.escala=preprocessing.MinMaxScaler() #Escala
            self.conj_entr=self.escala.fit_transform(self.conj_entr) #Aplicaci??n
        elif(metodo=="maxabs"): #Escalado entre 0 y 1
            self.escala=preprocessing.MaxAbsScaler() #Escala
            self.conj_entr=self.escala.fit_transform(self.conj_entr) #Aplicaci??n
    def kmeans(self,num_cum):
        self.kmedias=KMeans(n_clusters=num_cum).fit(self.conj_entr) #Entrenamiento
    def kmedians(self,num_cum):
        m=self.conj_entr.shape[1]
        centroides=[]
        for i in range(num_cum):
            arreglo=np.random.rand(m)
            centroides.append(arreglo)
        self.kmedianas=kmedians(self.conj_entr,centroides) #Inicializaci??n del algoritmo con el conjunto de entrenamiento y el n??mero de c??mulos deseados
        self.kmedianas.process() #Entrenamiento
    def kmedoids(self,num_cum): #Aun no aplicable, pide mucha memoria
        self.kmedoides=KMedoids(n_clusters=num_cum,method="pam",init="k-medoids++").fit(self.conj_entr) #Entrenamiento
    def som(self,x,y,porcentaje_base=0.15,vecindad=1.,aprendizaje=1.,topologia="hexagonal"):
        m=self.conj_entr.shape[1] #N??mero de variables en el conjunto de entrenamiento
        rango=np.arange(self.nn)
        cuantizacion=[]
        topologico=[]
        eje=[]
        self.mao=MiniSom(x,y,m,sigma=vecindad,learning_rate=aprendizaje,topology=topologia) #Inicializaci??n del algoritmo
#        self.mao.pca_weights_init(self.conj_entr)
        self.mao.random_weights_init(self.conj_entr) #Inicializaci??n de los pesos de las neuronas
        elementos=random.sample(sorted(rango),int(self.nn*porcentaje_base))
        conjunto=self.conj_entr[elementos,:]
        print()
        if(topologia=="rectangular"):
            for j in range(conjunto.shape[0]):
                print("\r Iteraci??n: "+str(j+1)+" de "+str(conjunto.shape[0]),end='')
                self.mao.update(conjunto[j],self.mao.winner(conjunto[j]),j,conjunto.shape[0])
                if(j%10000==0):
                    cuantizacion.append(self.mao.quantization_error(conjunto))
                    topologico.append(self.mao.topographic_error(conjunto)) #C??lculo del error topol??gico para redes rectangulares, se usa la implementada en la biblioteca minisom
                    eje.append(j)
            plt.pcolor(self.mao.distance_map(),cmap="autumn_r") #Graficaci??n de la matriz de distancias
            plt.colorbar()
            plt.title("U-matrix")
            plt.savefig("SOM.png")
            plt.clf()
        elif(topologia=="hexagonal"):
            for j in range(conjunto.shape[0]):
                print("\r Iteraci??n: "+str(j+1)+" de "+str(conjunto.shape[0]),end='')
                self.mao.update(conjunto[j],self.mao.winner(conjunto[j]),j,conjunto.shape[0])
                if(j%10000==0):
                    cuantizacion.append(self.mao.quantization_error(conjunto))
                    topologico.append(error_topologico_hexagonal(conjunto,self.mao)) #C??lculo del error tolol??gico para redes hexagonales, se usa implementaci??n propia
                    eje.append(j)
            xx, yy=self.mao.get_euclidean_coordinates() #A partir de aqu?? se hace la graficaci??n de la matriz de distancias, algoritmo tomado de la documentaci??n de la biblioteca minisom
            umatrix=self.mao.distance_map()
            weights=self.mao.get_weights()
            f=plt.figure(figsize=(10,10))
            ax=f.add_subplot(111)
            ax.set_aspect('equal')
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy=yy[(i,j)]*np.sqrt(3)/2
                    hexa=RegularPolygon((xx[(i,j)],wy),numVertices=6,radius=.95/np.sqrt(3),facecolor=plt.cm.autumn_r(umatrix[i,j]),edgecolor='black')
                    ax.add_patch(hexa)
            xran=np.arange(weights.shape[0])
            yran=np.arange(weights.shape[1])
            plt.xticks(xran-.5,xran)
            plt.yticks(yran*np.sqrt(3)/2,yran)
            divider=make_axes_locatable(plt.gca())
            ax_cb=divider.new_horizontal(size="5%",pad=0.05)
            cb1=colorbar.ColorbarBase(ax_cb,cmap=cm.autumn_r,orientation='vertical')
            cb1.ax.get_yaxis().labelpad=int(7*7)
            cb1.ax.set_ylabel('Distancia de las neuronas con sus vecinos',rotation=270, fontsize=16)
            plt.gcf().add_axes(ax_cb)
            plt.savefig('SOM.png')
            plt.clf()
        print("\n")
        plt.plot(eje,cuantizacion,label="Error de cuantizaci??n") #Graficaci??n de los errores de cuantizaci??n y topol??gico en el mismo gr??fico
        plt.plot(eje,topologico,label="Error topol??gico")
        plt.grid(True)
        plt.grid(ls='-',lw=0.25)
        plt.ylabel("Error")
        plt.xlabel("Iteraci??n")
        plt.title("Gr??fico de errores, mapa auto-organizado")
        plt.legend()
        plt.savefig("SOM-errores.png")
        plt.clf()
