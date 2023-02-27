import vtk
import random
import subprocess
import numpy as np
from resipy import Project
from scipy.interpolate import griddata
from herramientas import cercanos

########################################################
#                                                      #
# This class prepares the databases for further        #
# training and clustering, made for magnatometry and   #
# ERT2D surveys.                                       #
#                                                      #
# preparar(carpeta), where "carpeta" is the folder     #
# where the data files are stored.                     #
#                                                      #
# preparar.crear_base_magnetometria(vectores), where   #
# "vectores" are the stored variables in the files     #
# named as them. Is a list variable.                   #
#                                                      #
# preparar.crear_base_electrica(carpetas), where       #
# "carpetas" is a list with the respective inversion   #
# models folders from the resipy API.                  #
#                                                      #
# preparar.vecindad(vecinos), where "vecinos" is the   #
# number of neighbors elements to use in the training. #
#                                                      #
# preparar.guardar(nombre), where "nombre" is the      #
# output filename with the database preprocessed.      #
#                                                      #
########################################################

class preparar:
    def __init__(self,carpeta):
        self.carp=carpeta #Inicialización de la carpeta de trabajo
    def crear_base_magnetometria(self,vectores):
        carpeta=self.carp
        for i in range(len(vectores)): #Ciclo que desplaza cada elemento de la lista "vectores"
            if(i==0): #Primer ciclo, se almacenan las coordenadas únicamente
                d=open(carpeta+vectores[i],"r") #Se abre el archivo localizado en la carpeta de trabajo
                datos=np.genfromtxt(d,delimiter=",",skip_header=1)
                n=datos.shape[0] #Tamaño de la base de datos
                piv=np.zeros((n,1))
                self.variables=np.zeros((n,0))
                piv[:,0]=datos[:,0]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,1]
                self.variables=np.append(self.variables,piv,axis=1)
                d.close()
            else:
                d=open(carpeta+vectores[i],"r")
                datos=np.genfromtxt(d,delimiter=",",skip_header=1)
                piv[:,0]=datos[:,2]
                self.variables=np.append(self.variables,piv,axis=1) #Se agregan como columna cada variable ingresada por nombre de archivo
                d.close()
        nans=np.zeros(n) #Se inicializa un arreglo auxiliar para el manejo de valores no numéricos
        np.isnan(self.variables[:,2],nans) #Se eliminan los vectores que no poseen un valor numérico
        nans=1-nans
        nans=np.array(nans,dtype=bool)
        self.variables=self.variables[nans,:]
    def crear_base_electrica(self,carpetas):
        carpeta=self.carp
        cadena="/wd/f001_res.dat" #Se usa la dirección de las inversiones realizadas en resipy
        for i in range(len(carpetas)):
            d=open(carpeta+carpetas[i]+cadena,"r")
            datos=np.genfromtxt(d,delimiter="   ")
            if(i==0): #Se inicializa con el primer archivo
                n=datos.shape[0]
                piv=np.zeros((n,1))
                self.variables=np.zeros((n,0))
                piv[:,0]=datos[:,0]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,1]
                self.variables=np.append(self.variables,piv,axis=1)
                piv[:,0]=datos[:,2]
                self.variables=np.append(self.variables,piv,axis=1)
            else:
                z=griddata((datos[:,0],datos[:,1]),datos[:,2],(self.variables[:,0],self.variables[:,1]),method="linear") #Se interpolan los datos usando los puntos del primer vector
                piv[:,0]=z
                self.variables=np.append(self.variables,piv,axis=1)
            d.close()
    def vecindad(self,vecinos):
        n=self.variables.shape[0] #Tamaño de la base de datos
        m=self.variables.shape[1]-2 #Número de variables explicativas
        pivote=self.variables[:,0:2] #Arreglo auxiliar con las coordenadas x,y o x,z
        piv=np.zeros((n,1)) #Arreglo auxiliar de tamaño igual al tamaño de la base de datos
        distancias=np.zeros((n,vecinos+1)) #Arreglo en el que se almacenarán las distancias entre cada vector con coordenadas x,y o x,z
        for i in range(n):
            indices=cercanos(pivote[i,:],pivote) #Se calculan los vectores más cercanos en cuanto a posición espacial
            distancias[i,:]=indices[0:vecinos+1] #Los resultados de la linea anterior se almacenan en un arreglo (índices)
        for i in range(m):
            j=i+2 #Se itera sobre las variables explicativas, se excluyen coordenadas
            piv[:,0]=self.variables[:,j] #Se inicializa una arreglo auxiliar igual a la variable explicativa j de cada vector
            pivote=np.append(pivote,piv,axis=1) #Se agrega la variable explicativa al arreglo auxiliar que almacenaba las coordenadas
            for k in range(vecinos):
                for ii in range(n):
                    piv[ii,0]=self.variables[int(distancias[ii,k+1]),j] #Se identifican los k vecinos más próximos de cada punto
                pivote=np.append(pivote,piv,axis=1) #Se agrega la variable explicativa correspondiente al arreglo auxiliar
        self.variables=pivote #Se iguala el arreglo auxiliar, con la información de los vecinos, al arreglo de salida
    def transformar(self,archivo=None,metodo="ecualizar",bins='auto'):
        if("preparar.variables" in locals()):
            self.variables=self.variables
        else:
            d=open(self.carp+archivo)
            self.variables=np.genfromtxt(d,delimiter=",")
            d.close()
        if(metodo=="ecualizar"):
            for i in range(self.variables.shape[1]):
                if(i>1):
                    ecualizada=self.variables[:,i]
                    histograma,cajones=np.histogram(ecualizada,bins=bins)
                    maximo=max(ecualizada)
                    for j in range(ecualizada.shape[0]):
                        for k in range(cajones.shape[0]-1):
                            if(ecualizada[j]>=cajones[k] and ecualizada[j]<=cajones[k+1]):
                                break
                        suma=0
                        for l in range(k):
                            suma=suma+(histograma[l]/ecualizada.shape[0])
                        ecualizada[j]=suma*maximo
                    self.variables[:,i]=ecualizada
        elif(metodo=="logaritmica"):
            for i in range(self.variables.shape[1]):
                if(i>1):
                    transformada=self.variables[:,i]
                    for j in range(transformada.shape[0]):
                        transformada[j]=np.log10(transformada[j])
                    self.variables[:,i]=transformada
    def guardar(self,nombre):
        m=self.variables.shape[1] #Número de variables de la base de datos
        formato=[]
        for i in range(m):
            formato.append("%0.8f")
        np.savetxt(nombre,self.variables,delimiter=",",fmt=formato) #Almacenamiento

######################################################################
#                                                                    #
# This class performs a series of semi-automatic geoelectrical       #
# inverse models for the making of a training set.                   #
#                                                                    #
# modelar_electrica_2D(archivo,mallai,mallaf), where "archivo" is    #
# the file path of a base VTK mesh for the inverse models creation   #
# "mallai" is the file path with the fist section of the VTK mesh    #
# until the SCALARS res0 variable and LOOKUP_TABLE. "mallaf" is the  #
# file path with the last section of the VTK mesh until the SCALARS  #
# phase0 variable                                                    #
#                                                                    #
# modelar_electrica_2D.geometria(nelectrodos,dx),                    #
# where "nelectrodos" is the electrode number for direct modelling   #
# with a "dx" spacing.                                               #
#                                                                    #
# modelar_electrica_2D.modelo(secuencia,nelectrodos,dx), where       #
# "secuencia" is the file path which contains the automatic sequence #
# file whit headers C+,C-,P+,P-, "nelectrodos" is the electrode      #
# number in the sequence files and "dx" the spacing.                 #
#                                                                    #
# modelar_electrica_2D.modelos(intervalos,secuencias,n_modelos),     #
# where "intervalos" is a two-dimensional list variable with the     #
# bottom and upper limits for the creation of random resistivity     #
# values, "secuencias" is a list variable with the automatic         #
# sequence path to create the direct models, "n_modelos" is the      #
# number of models to create.                                        #
#                                                                    #
# modelar_electrica_2D.agrupar_secuencias(vecinos,iteracion,isecuencia), #
# where "vecinos" is the number of neighbors to add to the           #
# database's variables, "iteracion" is the number of inverse models  #
# to create, "isecuencia" is the number of automatic sequences       #
# applied                                                            #
#                                                                    #
# modelar_electrica_2D.agrupar_modelo(iteracion,isecuencia,vecinos), #
# where "iteracion", "isecuencia" and "vecinos" have the same        #
# variable behaviour of "agrupar_secuencias", the output database is #
# saved in a file named as "Modelos.dat".                            #
#                                                                    #
# ####################################################################

class modelar_electrica_2D:
    def __init__(self,archivo,mallai,mallaf):
        lector=vtk.vtkDataSetReader() #Inicialización del lector de archivos vtk
        lector.SetFileName(archivo) #Se carga el archivo vtk de entrada
        lector.ReadAllScalarsOn() #Lectura de los datos escalares (resistividad y región son las que no sinteresan)
        lector.Update()
        datos=lector.GetOutput() #Se guardan los datos en la variable
        celdas=datos.GetCellData() #En celdas se almacena la información de datos
        reg=celdas.GetArray('region') #En reg se guardan los datos de region
        self.nd=datos.GetNumberOfCells() #Almacenamiento del número de celdas
        self.regiones=np.array([reg.GetValue(i) for i in range(self.nd)]) #Conversión de los valores de región a arreglo de numpy, con el fin de cotejar la región de cada celda con su valor de resistividad
        self.nreg=np.unique(self.regiones) #Obtención del número de regiones
        print("\n",len(self.nreg),"regiones\n")
        with open(mallai,"r") as archivo:
            self.inicio=archivo.read() #Se carga la primer parte del archivo vtk
        with open(mallaf,"r") as archivo:
            self.fin=archivo.read() #Se carga la segunda parte del archivo vtk, se espera únicamente agregar valores de resistividad para generar un nuevo archivo
    def geometria(self,nelectrodos,dx):
        self.electrodos=np.zeros((nelectrodos,3)) #Inicialización de la geometría de los electrodos
        self.electrodos[:,0]=np.arange(0,nelectrodos*dx,dx) #Posición de los electrodos
    def modelo(self,secuencia,nelectrodos,dx):
        proyecto=Project(dirname="tmp",typ="R2") #Inicialización del proyecto en resipy
        self.geometria(nelectrodos,dx)
        proyecto.setElec(self.electrodos) #Definición de electrodos
        proyecto.importMesh("modelar.vtk") #Importación de Malla a usar para la inversión
        proyecto.importSequence(secuencia) #Se importa la secuencia a usar para la inversión
        proyecto.forward(noise=0.0) #Modelación directa sin ruido
        proyecto.invert() #Inversión del modelo directo creado
        d=open("tmp/invdir/f001_res.dat") #Se abre el archivo generado
        self.rho=np.genfromtxt(d) #Almacenamiento de la resistividad
        d.close()
    def modelos(self,intervalos,secuencias,n_modelos,nelectrodos,dx):
        resistividades=np.zeros(self.nd) #Inicialización de arreglo para almacenar resistividades
        self.nmodelos=n_modelos
        self.nsecuencias=len(secuencias)
        for ijk in range(n_modelos): #Ciclo que itera sobre el número de modelos
            for region in self.nreg: #Ciclo que itera sobre el número de regiones
                for i in range(self.nd): #Ciclo que itera sobre el número de celdas
                    if(self.regiones[i]==region): #Si la región de la celda es igual a la región iterada
                        resistividades[i]=random.uniform(intervalos[int(region)-1][0],intervalos[int(region)-1][1]) #Se calcula una resistividad aleatoria dentro de un intervalo
            cadena=np.array2string(resistividades,precision=8,separator="     ",threshold=self.nd+1,max_line_width=self.nd+1) #El arreglo de resistividades se convierte a cadena
            with open("modelar.vtk","w") as archivo:
                archivo.write(self.inicio+cadena[1:-1]+self.fin) #Se almacena un archivo vtk con las resistividades calculadas y el inicio y fin de un archivo vtk
            isecuencia=-1 #Inicialización de número auxiliar de secuencias
            for secuencia in secuencias: #Ciclo que itera sobre la cantidad de secuencias
                isecuencia=isecuencia+1
                self.modelo(secuencia,nelectrodos,dx) #Se realiza la modelación usando la secuencia en cuestión
                self.rho=np.delete(self.rho,3,1) #Se elimina la columna con logaritmo
                np.savetxt("Secuencia-"+str(isecuencia)+"_modelo-"+str(ijk),self.rho,delimiter=', ',header="X, Y, "+str(isecuencia),fmt=['%0.8f','%0.8f','%0.8f']) #Almacenamiento de los resultados del modelo y secuencia actual
            subprocess.run(["rm","modelar.vtk"]) #Eliminación del archivo vtk usado para modelar
    def agrupar_secuencias(self,vecinos,iteracion=0,isecuencia=0):
        if(iteracion==0 and isecuencia==0): #Carga de los números de modelos y secuencias usados, con el fin de usar datos de una ejecución independiente
            iteracion=self.nmodelos
            isecuencia=self.nsecuencias
        self.vec=vecinos
        for i in range(iteracion): #Ciclo que itera sobre el número de modelos
            formato=[]
            for j in range(isecuencia): #Ciclo que itera sobre el número de secuencias
                d=open("Secuencia-"+str(j)+"_modelo-"+str(i)) #Apertura del archivo de acuerdo con el número de modelo y número de secuencia
                datos=np.genfromtxt(d,delimiter=",",skip_header=1) #Los datos dek archivo se almacenan
                d.close()
                n=datos.shape[0] #Tamaño de la base de datos actual
                if(j==0): #Si es la primer secuencia se inicializan los arreglos
                    datos_secuencia=np.zeros((n,0)) #Inicialización de variable para almacenar los datos de las secuencias para un modelo fijo
                    datos_secuencia=np.append(datos_secuencia,datos[:,0:2],axis=1) #Se agregan los datos espaciales
                    formato.append('%0.8f')
                    formato.append('%0.8f')
                pivote=np.zeros((n,1)) #Inicialización de arreglo auxiliar...
                pivote[:,0]=datos[:,2] #... que se iguala a los datos de resistividad almacenados anteriormente
                datos_secuencia=np.append(datos_secuencia,pivote,axis=1) #Se agregan los datos de resistividad al arreglo de salida
                formato.append('%0.8f')
                if(vecinos>0): #Si se desea agregar información sobre los puntos vecinos:
                    datos_secuencia=np.append(datos_secuencia,np.zeros((n,vecinos)),axis=1) #Se agrega a la variable de salida un arreglo de tamaño igual al número de puntos vecinos deseados
                    for k in range(vecinos):
                        formato.append('%0.8f')
                    piv=datos_secuencia[:,0:2] #Arreglo auxiliar que almacena las coordenadas de cada vector
                    distancias=np.zeros((n,n)) #Inicialización de matriz de "distancias"
                    for k in range(n): #Ciclo que itera sobre el número de vectores, se obtienen los indices de los vectores más cercanos en orden ascendente
                        indices=cercanos(piv[k,:],piv)
                        distancias[k,:]=indices
                    for k in range(vecinos): #Ciclo que itera sobre el número de vecinos
                        for l in range(n): #Ciclo que itera sobre el número de vectores
                            datos_secuencia[l,datos_secuencia.shape[1]-int(vecinos)+k]=datos[int(distancias[l,k+1]),2] #Se iguala cada vecino de cada vector al valor de resistividad correspondiente
                subprocess.run(["rm","Secuencia-"+str(j)+"_modelo-"+str(i)]) #Se elimina el archivo leido
            np.savetxt("Modelo-"+str(i),datos_secuencia,delimiter=', ',fmt=formato) #Almacenamiento de un nuevo archivo con las secuencias agrupadas a forma de columna
    def agrupar_modelo(self,iteracion=0,isecuencia=0,vecinos=0):
        if(iteracion==0 and isecuencia==0 and vecinos==0): #Carga de los números de modelos, secuencias y vecinos usados, con el fin de usar datos de una ejecución independiente
            iteracion=self.nmodelos
            isecuencia=self.nsecuencias
            vecinos=self.vec
        formato=[]
        j=(int(vecinos)+1)*int(isecuencia)+2 #Cálculo del número de columnas que se obtendrá, depende del número de vecinos y de las secuencias realizadas (coordenadas también)
        entrenamiento=np.zeros((0,j)) #Inicialización de la base de datos final
        for i in range(j):
            formato.append('%0.8f')
        for i in range(iteracion): #Ciclo que itera sobre el número de modelos
            d=open("Modelo-"+str(i))
            datos=np.genfromtxt(d,delimiter=",") #Se lee cada archivo de modelo realizado
            entrenamiento=np.append(entrenamiento,datos,axis=0) #Se agrega cada elemento de todos los modelos como un nuevo vector
            d.close()
        np.savetxt("Modelos.dat",entrenamiento,delimiter=', ',fmt=formato) #Almacenamiento de la base de datos
