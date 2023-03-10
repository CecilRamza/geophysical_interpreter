import vtk
import math
import colorsys
import numpy as np
import pyvista as pv
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
import sklearn.manifold as manifold
from matplotlib import cm, colorbar
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from matplotlib.patches import RegularPolygon
from pyclustering.cluster.kmedians import kmedians

#########################################################
#                                                       #
# This class aplies the clustering methodology on the   #
# clustering set using the output parameters from the   #
# training step. The grids are prepared to display the  #
# SOM or clusters, as well as the display of the        #
# automated interpretation of geophysical data.         #
# entrenamiento.clasificar stores the clustered data.   #
#                                                       #
# clasificacion(archivo,columnas), where "archivo" is   #
# the file path to the database to apply the clustering,#
# "columnas" is a list with the column numbers of the   #
# variables to use to cluster, this variables needs to  #
# be the ones chosen in the training step.              #
# entrenamiento. clasificacion.x are the x coordinates, #
# clasificacion.y are the y coordinates,                #
# clasificacion.nn is the database size [0].            #
#                                                       #
# clasificacion.reduccion_pca(columnas,estandarizaciones,pcas,componentes), #
# where "columnas" is a list that contains the variables#
# set to reduce its dimensionality, this sets needs to  #
# be the same used in the training step.                #
# "estandarizaciones" are the transformations used      #
# to standarize the training set. "pcas" contains the   #
# transformation matrix of each variable set obtained   #
# from the training step.                               #
# "componentes" is a list with the number of components #
# to reduce the dimensionality using the PCA algorithm, #
# this list needs to be the same used in the training   #
# step.                                                 #
#                                                       #
# clasificacion.escalamiento(escala), where "escala"    #
# is the transformation used in the training step.      #
#                                                       #
# clasificacion.kmeans(kmeans), where "kmeans" is the   #
# output object from the training step.                 #
# clasificacion.etiquetas are the resulting data labels.#
# clasificacion.centros cluster coordinates. This method#
# generates an histogram showing the using frequency    #
# of each cluster.                                      #
#                                                       #
# clasificacion.kmedians(kmedians), where "kmedians"    #
# is the output object from the training step.          #
# clasificacion.etiquetas are the resulting data labels.#
# clasificacion.centros are the clusters coordinates    #
# This method generates an histogram showing the using  #
# frequency of each cluster.                            #
#                                                       #
# clasificacion.som(som), where "som" is the resulting  #
# map from the clustering. "clasificacion.indices is an #
# array with neuron's labels,                           #
# clasificacion.etiquetas are the data labels,          #
# clasificacion.pesos are the neuron's weights,         #
# clasificacion.nx is the number of rows of the build   #
# network, clasificacion.ny is the number of columns of #
# the build network. This method generates an histogram #
# showing the using frequency of each neuron.           #
#                                                       #
# clasificacion.etiquetar(columna), adds the            #
# coordinates, value of variables and label (RGB) value #
#to the object clasificacion.clasificados.              #
#                                                       #
# clasificacion.malla(metodo,radio,cumulos,metodo_sub). #
# Where "metodo" is the used in the training and        #
# clustering set: "centroides" (for k-means and         #
# k-medians), "red_rectangular" or "red_hexagonal" for  #
# the SOM algorithm. "radio" are the sphere ratios to   #
# display the clusters when the "centroides" method is  #
# chosen. "cumulos" is the number of neurons culsters   #
# to compute the color vector for each one. "metodo_sub"#
# is the method to generate the neurons clusters:       #
# "kmedias", "kmedianas" o "kmedoides".                 #
# clasificacion.malla_etiquetas contains the grid ([0]) #
# and the color vector of each cell ([1]).              #
#                                                       #
#########################################################

class clasificacion:
    def __init__(self,archivo,columnas,escalamiento_min=[],escalamiento_max=[]):
        d=open(archivo)
        self.originales=np.genfromtxt(d,delimiter=",") #Se almacenan los datos originales
        d.close()
        d=open(archivo)
        self.clasificar=np.genfromtxt(d,delimiter=",") #Lectura de los datos
        d.close()
        self.x=self.clasificar[:,0]
        self.x=np.reshape(self.x,(self.x.shape[0],1)) #Se almacena la coordenada en x
        self.y=self.clasificar[:,1]
        self.y=np.reshape(self.y,(self.y.shape[0],1)) #Se almacena la coordenada en y
        n=self.clasificar.shape[1] #N??mero de variables explicativas
        mascara=np.zeros(n,dtype=bool)
        mascara[columnas]=True
        self.clasificar=self.clasificar[:,mascara] #Se separan las variables explicativas (columnas) que se van a usar
        self.nn=self.clasificar.shape[0] #Tama??o de la base de datos
        n=self.clasificar.shape[1] #N??mero de variables explicativas
        if(len(escalamiento_min)>0 and len(escalamiento_max)>0 and len(escalamiento_min)==len(escalamiento_max)):
            for i in range(n):
                pivote=self.clasificar[:,i]
                rmin=min(pivote)
                rmax=max(pivote)
                for j in range(pivote.shape[0]):
                    pivote[j]=(pivote[j]-rmin)/(rmax-rmin)*(escalamiento_max[i]-escalamiento_min[i])+escalamiento_min[i]
                self.clasificar[:,i]=pivote
    def reduccion_pca(self,columnas,estandarizaciones,pcas,componentes):
        sin_cambios=np.arange(self.clasificar.shape[1]) #Variables explicativas originales
        npo=len(pcas) #N??mero de componentes principales
        for columna in columnas:
            for digito in columna:
                if digito in sin_cambios:
                    sin_cambios=np.delete(sin_cambios,np.where(sin_cambios==digito)[0]) #Variables a las que no se aplicar?? pca
        conjunto=self.clasificar[:,sin_cambios] #Conjunto con las variables que no se reducir??n
        for i in range(npo):
            arreglo=estandarizaciones[i].transform(self.clasificar[:,columnas[i]]) #Estandarizaci??n de las variables
            arreglo=np.matmul(arreglo,np.transpose(pcas[i].components_[0:int(componentes[i]),:])) #Aplicaci??n de pca
            arr=arreglo.reshape((self.nn,int(componentes[i])))
            conjunto=np.append(conjunto,arr,axis=1) #Se agregan los datos reducidos dimensionalmente
        self.clasificar=conjunto #Se obtiene el conjunto reducido dimensionalmente
    def escalamiento(self,escala):
        self.clasificar=escala.transform(self.clasificar) #Escalamiento de la base de datos
    def malla(self,metodo="centroides",radio=1.,cumulos=2,metodo_sub="kmedoides"):
        if(metodo=="centroides"):
            nc=self.centros.shape[0] #N??mero de centroides
            iso=manifold.Isomap(n_neighbors=2,n_components=3) #Reducci??n a dimensionalidad 3 usando isomap
            iso.fit(self.centros)
            self.centros=iso.transform(self.centros)
            colores=np.zeros((nc,3)) #Arreglo para almacenar los colores
            etiquetas2d=np.zeros((0,3)) #Arreglo que define el color para cada celda
            pivote=np.zeros((1680,3)) #Arreglo auxiliar que almacena las celdas de una esfera
            dh=1/(nc+0.01) #Delta de matiz
            h=-dh #Inicializaci??n de el matiz
            radios=np.zeros(nc)
            for i in range (nc):
                for j in range(self.nn):
                    if(int(self.etiquetas[j,0])==i):
                        radios[i]=radios[i]+1
            minimo=min(radios)
            maximo=max(radios)
            for i in range(nc):
                radios[i]=((radios[i]-minimo)/(maximo-minimo)*9.)+1.
                if(radios[i]==1):
                    radios[i]=1.0001
                radios[i]=math.log10(radios[i])*radio
            for i in range(nc):
                if(i==0):
                    plano=pv.Sphere(center=(self.centros[i,0],self.centros[i,1],self.centros[i,2]),radius=radios[i]) #Inicializaci??n de la malla con el primer centroide
                else:
                    esfera=pv.Sphere(center=(self.centros[i,0],self.centros[i,1],self.centros[i,2]),radius=radios[i]) #Esfera del siguiente centroide
                    plano=plano+esfera #Se agrega la esfera a la malla
                h=h+dh #Incremento en el matiz
                rgb=np.array(colorsys.hsv_to_rgb(h,0.75,0.9)) #Obtenici??n del color rgb con la matiz, siendo la intensidad y satuzaci??n fijos
                colores[i,0]=round(rgb[0]*255) #Canal rojo
                colores[i,1]=round(rgb[1]*255) #Canal verde
                colores[i,2]=round(rgb[2]*255) #Canal azul
                pivote[:,:]=colores[i,:] #Se iguala el color a la esfera pivote
                etiquetas2d=np.vstack((etiquetas2d,pivote)) #Se agregan las celdas de la esfera actual
            pivote=np.zeros((self.nn,3)) #Arreglo pivote para definir las nuevas etiquetas
            for i in range(self.nn):
                pivote[i,:]=colores[int(self.etiquetas[i,0]),:] #Obtenici??n de las etiquetas con color
            self.etiquetas=pivote #Nuevas etiquetas
        elif(metodo=="red_hexagonal" or metodo=="red_rectangular"):
            if(metodo_sub=="kmedias"): #Se realiza la aglomeraci??n de las neuronas para definir subconjuntos
                etiquetas=[] #Inicializaci??n de arreglo para almacenar las neuronas que se parecen entre s??
                kmedias=KMeans(n_clusters=cumulos).fit(self.pesos) #Aplicaci??n de k-medias
                etiquetasp=kmedias.labels_ #Se almacenan las etiquetas obtenidas
                for i in range(cumulos):
                    cumulo=[] #Se inicializa un arreglo para almacenar las neuronas que pertenecen al mismo centroide
                    for j in range(self.pesos.shape[0]):
                        if(etiquetasp[j]==i):
                            cumulo.append(j) #Se agrega la neurona al arreglo del centroide actual
                    etiquetas.append(cumulo) #Se agrega el conjunto de neuronas del centroide actual al arreglo de etiquetas
            elif(metodo_sub=="kmedianas"):
                columnas=self.pesos.shape[1]
                centros=[]
                for i in range(cumulos):
                    arreglo=np.random.rand(columnas)
                    centros.append(arreglo)
                kmedianas=kmedians(self.pesos,centros)
                kmedianas.process()
                etiquetas=kmedianas.get_clusters()
            elif(metodo_sub=="kmedoides"):
                etiquetas=[]
                kmedoides=KMedoids(n_clusters=cumulos,method="pam",init="k-medoids++").fit(self.pesos)
                etiquetasp=kmedoides.labels_
                for i in range(cumulos):
                    cumulo=[]
                    for j in range(self.pesos.shape[0]):
                        if(etiquetasp[j]==i):
                            cumulo.append(j)
                    etiquetas.append(cumulo)
            etiquetas2d=np.zeros((self.nx*self.ny,3)) #Se inicializa arreglo para almacenar los colores de cada celda
            dh=1/(cumulos+0.01) #Delta de matiz
            h=dh/2 #Inicializaci??n del matiz
            for cumulo in etiquetas: #Ciclo que itera sobre cada centroide
                dc=0.9/len(cumulo) #Delta saturaci??n
                h=h+dh #Actualizaci??n de matiz
                s=1 #Inicializaci??n de saturaci??n
                for elemento in cumulo: #Ciclo que itera sobre las neuronas que pertenecen al centroide actual
                    s=s-dc #Actualizaci??n de saturaci??n
                    v=1 #valor de intensidad
                    rgb=np.array(colorsys.hsv_to_rgb(h,s,v)) #Conversi??n de HSI a RGB
                    etiquetas2d[int(elemento),0]=round(rgb[0]*255) #Canal rojo
                    etiquetas2d[int(elemento),1]=round(rgb[1]*255) #Canal verde
                    etiquetas2d[int(elemento),2]=round(rgb[2]*255) #Canal azul
            pivote=np.zeros((self.nn,3)) #Arreglo auxiliar para definir las nuevas etiquetas
            for i in range(self.nn):
                pivote[i,:]=etiquetas2d[int(self.etiquetas[i,0]),:] #Obtenici??n de las etiquetas con color
            self.etiquetas=pivote #Nuevas etiquetas
            if(metodo=="red_hexagonal"):
                hexagonop=np.array([[0,0,0.1],[0,-1,0.1],[0.866,-1.5,0.1],[1.732,-1,0.1],[1.732,0,0.1],[0.866,0.5,0.1],[0,0,0],[0,-1,0],[0.866,-1.5,0],[1.732,-1,0],[1.732,0,0],[0.866,0.5,0]]) #Hex??gono base
                hexagono=np.zeros((hexagonop.shape[0],hexagonop.shape[1])) #Arreglo para almacenar nuevas coordenadas de hex??gonos desplazados
                hexagono[:,:]=hexagonop[:,:] #Inicializaci??n de arreglo de hex??gono m??vil
                puntos=np.empty((0,3)) #Inicializaci??n del arreglo que almacenar?? los v??rtices de todos los hex??gonos
                for i in range(self.nx): #Ciclo que itera sobre la direcci??n vertical de la red neuronal
                    if(i>0): #Cuando no se trata de la primera fila
                        if(i%2==1): #Si se trata de una fila impar
                            hexagonop[:,0]=hexagonop[:,0]+0.866 #Se desplazan las coordenadas horizontales hacia la derecha
                            hexagonop[:,1]=hexagonop[:,1]-1.5 #Se desplazan las coordenadas verticales
                            hexagono[:,:]=hexagonop[:,:] #El nuevo hex??gono que se desplazar??
                        else: #Si se trata de una fila par
                            hexagonop[:,0]=hexagonop[:,0]-0.866 #Se desplazan las coordenadas horizontales hacia la izquierda
                            hexagonop[:,1]=hexagonop[:,1]-1.5 #Se desplazan las coordenadas verticales
                            hexagono[:,:]=hexagonop[:,:] #El nuevo hex??gono que se desplazar??
                    for j in range(self.ny): #Ciclo que itera sobre la direcci??n horizontal de la red neuronal
                        if(i==0 and j==0): #Si es la primer neurona
                            arreglo=np.arange(12) #N??mero de puntos que formar??n al hex??gono
                            celdas=np.hstack(([12],arreglo)) #Se agrega la celda
                            tipo=np.array([vtk.VTK_HEXAGONAL_PRISM]) #Se agrega el tipo de celda
                        else: #Si no es la primer neurona
                            arreglo=arreglo+12 #Se obtiene los puntos de la siguiente neurona
                            celdas=np.hstack((celdas,[12])) #Se agrega la celda
                            celdas=np.hstack((celdas,arreglo))
                            tipo=np.hstack((tipo,[vtk.VTK_HEXAGONAL_PRISM])) #Se agrega el tipo de celda
                        puntos=np.vstack((puntos,hexagono)) #Se agregan los puntos de la celda
                        hexagono[:,0]=hexagono[:,0]+1.732 #Se desplaza horizontalmente la celda
                plano=pv.UnstructuredGrid(celdas,tipo,puntos) #Se define la malla no estructurada
            elif(metodo=="red_rectangular"):
                x=np.linspace(0,self.nx,self.nx+1) #N??mero de puntos en las celdas horizontales
                y=np.linspace(0,self.ny,self.ny+1) #N??mero de p??ntos en las celdas verticales
                x,y=np.meshgrid(y,x) #Mallado de puntos para coordenadas
                z=np.zeros((self.nx+1,self.ny+1)) #Se desea obtener una malla plana
                malla=pv.StructuredGrid(x,y,z) #Malla rectangular
                plano=malla.flip_y(inplace=False) #Se voltea la malla para cambiar la definici??n de celdas
        self.malla_etiquetas=[] #Arreglo para almacenar la malla y los valores de cada celda
        self.malla_etiquetas.append(plano)
        self.malla_etiquetas.append(etiquetas2d)
    def kmeans(self,kmeans):
        self.etiquetas=kmeans.predict(self.clasificar) #Obtenici??n de etiquetas
        self.etiquetas=np.reshape(self.etiquetas,(self.etiquetas.shape[0],1)) #Cambio a dimensi??n (n,1)
        self.centros=kmeans.cluster_centers_ #Coordenadas de los centroides
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
#                histograma[i,1]=math.log10(histograma[i,1])
                histograma[i,1]=histograma[i,1]
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-means")
        plt.savefig("Histograma.png")
        plt.clf()
    def kmedians(self,kmedians):
        self.etiquetas=np.zeros((self.nn,1)) #Inicializaci??n de las etiquetas
        self.centros=np.array(kmedians.get_medians()) #Coordenadas de los centroides
        for i in range(self.nn):
            indices=cercanos(self.clasificar[i,:],self.centros) #Se busca el centroide m??s cercano a cada dato
            self.etiquetas[i,0]=indices[0] #Se etiqueta al dato
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=math.log10(histograma[i,1])
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-medians")
        plt.savefig("Histograma.png")
        plt.clf()
    def kmedoids(self,kmedoids): #A??n no aplicable por cuestiones de memoria
        self.etiquetas=kmedoids.predict(self.clasificar) #Obtenci??n de las etiquetas
        self.etiquetas=np.reshape(self.etiquetas,(self.etiquetas.shape[0],1)) #Cambio a dimensi??n (n,1)
        self.centros=kmedoids.cluster_centers_ #Coordenadas de los centroides
        nc=self.centros.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=math.log10(histograma[i,1])
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de los centroides / k-medoids")
        plt.savefig("Histograma.png")
        plt.clf()
    def som(self,som,topologia="rectangular"):
        mapa=som.distance_map() #Obtenci??n del mapa de distancias
        self.indices=np.arange(mapa.shape[0]*mapa.shape[1]) #Se obtiene el n??mero de neuronas
        self.indices=np.reshape(self.indices,(mapa.shape[0],mapa.shape[1])) #Cambio a dimensi??n (n,m)
        self.etiquetas=np.zeros((self.nn,1)) #Inicializaci??n de etiquetas
        self.pesos=som.get_weights() #Obtenci??n de los pesos de las neuronas
        self.nx=self.pesos.shape[0] #N??mero de filas en la red
        self.ny=self.pesos.shape[1] #N??mero de columnas en la red
        self.pesos=np.reshape(self.pesos,(self.nx*self.ny,self.pesos.shape[2])) #Cambio a dimensiones para coincidir con la red
        for i in range(self.nn):
            neurona=som.winner(self.clasificar[i,:]) #Obtenci??n de la neurona ganadora
            self.etiquetas[i,0]=self.indices[int(neurona[0]),int(neurona[1])] #Se etiqueta al dato
        nc=self.indices.shape[0]
        histograma=np.zeros((nc,2))
        formato=["%d","%0.8f"]
        for i in range(nc):
            histograma[i,0]=i+1
            for j in range(self.nn):
                if(int(self.etiquetas[j,0])==i):
                    histograma[i,1]=histograma[i,1]+1
        for i in range(nc):
            if(histograma[i,1]==0):
                histograma[i,1]=0
            else:
                histograma[i,1]=histograma[i,1]
        np.savetxt("Histograma.dat",histograma,delimiter=" ",fmt=formato)
        n,bins,patches=plt.hist(x=self.etiquetas[:,0],bins="auto",color="#0504aa",alpha=0.7,rwidth=0.9)
        plt.grid(axis="y",alpha=0.5)
        plt.xlabel("Centroide")
        plt.ylabel("Frecuencia")
        plt.title("Histograma de las neuronas / SOM")
        plt.savefig("Histograma.png")
        plt.clf()
        if(topologia=="rectangular"):
            frecuencias=som.activation_response(self.clasificar)
            plt.pcolor(frecuencias,cmap="inferno_r")
            plt.colorbar()
            plt.title("Hit-map")
            plt.savefig("SOM_hits.png")
            plt.clf()
        elif(topologia=="hexagonal"):
            frecuencias=som.activation_response(self.clasificar)
            xx, yy=som.get_euclidean_coordinates() #A partir de aqu?? se hace la graficaci??n de la matriz de distancias, algoritmo tomado de la documentaci??n de la biblioteca minisom
            weights=som.get_weights()
            f=plt.figure(figsize=(10,10))
            ax=f.add_subplot(111)
            ax.set_aspect('equal')
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy=yy[(i,j)]*np.sqrt(3)/2
                    hexa=RegularPolygon((xx[(i,j)],wy),numVertices=6,radius=.95/np.sqrt(3),facecolor=plt.cm.inferno_r(frecuencias[i,j]),edgecolor='black')
                    ax.add_patch(hexa)
            xran=np.arange(weights.shape[0])
            yran=np.arange(weights.shape[1])
            plt.xticks(xran-.5,xran)
            plt.yticks(yran*np.sqrt(3)/2,yran)
            divider=make_axes_locatable(plt.gca())
            ax_cb=divider.new_horizontal(size="5%",pad=0.05)
            cb1=colorbar.ColorbarBase(ax_cb,cmap=cm.inferno_r,orientation='vertical')
            cb1.ax.get_yaxis().labelpad=int(7*7)
            cb1.ax.set_ylabel('N??mero de aciertos',rotation=270, fontsize=16)
            plt.gcf().add_axes(ax_cb)
            plt.savefig('SOM_hits.png')
            plt.clf()
    def etiquetar(self,columna):
        aux=self.originales[:,columna+2] #Arreglo auxiliar con la columna de la base original
        aux=np.reshape(aux,(aux.shape[0],1))
        self.clasificados=np.append(self.x,self.y,axis=1) #Se agrega al arreglo de salida las coordenadas x y y
        self.clasificados=np.append(self.clasificados,aux,axis=1) #Se agrega al arreglo de salida la columna de la base original
        escala=preprocessing.StandardScaler().fit(aux) #Escala
        aux=escala.transform(aux) #Aplicaci??n
#        aux=self.clasificar[:,columna] #Arreglo auxiliar con la columna de la base pre-procesada
        aux=np.reshape(aux,(aux.shape[0],1))
        self.clasificados=np.append(self.clasificados,aux,axis=1) #Se agrega al arreglo de salida la columna de la base pre-procesada
        self.clasificados=np.append(self.clasificados,self.etiquetas,axis=1) #Se agrega al arreglo de salida las etiquetas, esperando sean las etiquetas de color RGB.
