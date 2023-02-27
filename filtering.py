import pywt
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.ensemble import IsolationForest as IsFo
from sklearn.neighbors import LocalOutlierFactor as LOF
from tools import plot_signal_decomp

#######################################################################
#                                                                     #
# This class applies filtering algorithms on datasets with two        #
# different formats: ERT2D data and magnetometric data                #
# magnetométricos.                                                    #
#                                                                     #
# Uso sugerido:                                                       #
#                                                                     #
# ERT2D: after the initialization the user can choose between two     #
# methods.                                                            #
# filtrado(archivo,"eléctrica"), where "archivo" is the path to the   #
# database, the algorithm needs the "syscal" format file input, where #
# the first and second columns are the depth level and the X position #
# of the attribution point                                            #
# filtrado.ondicula_electrica(voltaje,ondicula,componentes,recursion) #
# where "voltaje" is the column number where is saved the desired     #
# variable to filter. "ondicula" defines the wavelet to use           #
# according to the pywt definitions. "componentes" is the number of   #
# decompositions that will be applied to the original signal.         #
# "recursion" is the number of recursions to filter each depth level. #
# In each recursion will be displayed the multiscale decomposition    #
# plot, next the system waits the input of attenuation factors for    #
# each detail component.                                              #
# filtrado.atipicos_electrica(logaritmo,metodo,vecinos,umbral)        #
# where "logaritmo" is a boolean variable to choose the application   #
# of a logarithm transformation. "metodo" can be "LOF"                #
# ("local outlier factor") or "IF" ("isolation forests"). "vecinos"   #
# is the number of neighbors to use for the "LOF" algorithm. "umbral",#
# for "LOF" will be the quantile to define an anomalous data; if "IF" #
# is chosen, every vector with anomaly score lower than "umbral" will #
# be labeled as anomaly.                                              #
# The resistivity variable needs be defined on the seventh column of  #
# the database.                                                       #
#                                                                     #
# Magnetometry: initialize and choose the method.                     #
# fitrado(archivo,"magnetometría"), where "archivo" is the path to    #
# the database, the first and second columns are the coordinates of   #
# each data point.                                                    #
# filtrado.ondicula_magnetometria(columnas,resolucion_x,resolucion_y,malla,ventana,ondicula,componentes,pesos,pesos2) #
# "colummnas" is a tuple with the column label to filter, begining    #
# with 0. "resolucion_x" y "resolucion_y" are the number of elements  #
# to interpolate to create the mesh. "malla" is a boolean variable    #
# used to specify the filter type: complete dataset, instead          #
# "ventana" is used to specify a windowed filter defined by a         #
# coordinate pair defined interactively by the user (upper left and   #
# bottom right coordinates). "ondicula" defines the wavelet to use    #
# according to the pywt definitios. "componentes" is the number of    #
# decompositions that will be applied to the original 2D signal.      #
# "pesos" are the components attenuation factors according to the     #
# DWT2D, starts with the approximation component before the ordered   #
# detail components. "pesos2" are the attenuation factors when        #
# "malla=True" and "ventana=True"                                     #
#                                                                     #
# filtrado.guardar(nombre), save the filtered data.                   #
#                                                                     #
#######################################################################

class filtrado:
    def __init__(self,archivo,tipo):
        self.tipo=tipo
        d=open(archivo)
        self.datos=np.genfromtxt(d,delimiter=",",skip_header=1)
        d.close()
        if(tipo=="eléctrica"):
            self.niveles=int(max(self.datos[:,0])) #Se define el número de niveles
        elif(tipo=="magnetometría"):
            self.equis=np.zeros(0) #Se inicializa la variable de las coordenadas x de la ventana que elige el usuario en una figura
            self.ye=np.zeros(0) #Se inicializa la variable de las coordenadas y de la ventana que elige el usuario en una figura
        self.nd=self.datos.shape[0] #Tamaño de la base de datos
        self.nnd=self.datos.shape[1] #Número de columnas de la base de datos
    def evento_mouse(self,event): #Se obtienen las coordenadas de un punto a partir del click en una figura
        self.equis=np.append(self.equis,event.xdata)
        self.ye=np.append(self.ye,event.ydata)
    def ondicula_electrica(self,voltaje,ondicula="db2",componentes=5,recursion=1):
        modo=pywt.Modes.periodization #Se usa periodización para la transformada wavelet
        self.filtrados=np.zeros((self.nd,self.nnd))
        self.filtrados=self.datos #Se inicializan los datos filtrados igual a los datos originales
        for i in range(self.niveles): #Desplazamiento de los niveles
            nivel=i+1
            arreglo=self.datos[np.where(nivel==self.datos[:,0])] #Se separan los datos del nivel actual
            filtrar=arreglo[:,voltaje] #Se separa la columna a filtrar
            n=filtrar.shape[0] #Tamaño del nivel
            for iteracion in range(recursion): #Recursión
                if(i==0):
                    plot_signal_decomp(filtrar,ondicula,"Nivel "+str(i+1)) #Gráfica del análisis en multiresolución
                    atenuacion=np.zeros(componentes) #A partir de aquí se convierten los elementos de la tupla a un arreglo de numpy
                    for j in range(componentes):
                        cadena=input("Ingrese los factores de atenuación de la componente "+str(j+1)+": ")
                        atenuacion[j]=float(cadena)
                    atenuacion=np.flipud(atenuacion)
                    print()
                potencia=2 #A partir de aquí se encuentra el menor número potencia de 2 que es mayor al tamaño del nivel
                while potencia<n:
                    potencia=potencia*2
                pivote=np.zeros(potencia)
                pivote[0:n]=filtrar #Arreglo a filtrar cuyo tamaño es potencia de 2
                detalles=[]
                for componente in range(componentes): #Aplicación de la transformada wavelet
                    aproximacion,d=pywt.dwt(pivote,ondicula,modo)
                    detalles.append(d)
                    pivote=aproximacion
                detalles.reverse()
                for componente in range(componentes): #Aplicación de los pesos
                    if(atenuacion[componente]!=0):
                        aproximacion=pywt.idwt(aproximacion,detalles[componente]*atenuacion[componente],ondicula,modo)
                    else:
                        print(i)
                        aproximacion=pywt.idwt(aproximacion,None,ondicula,modo)
                filtrar=aproximacion[0:n]
            pivote=np.zeros(n)
            pivote=filtrar
            self.filtrados[np.where(nivel==self.datos[:,0]),voltaje]=pivote #Se igualan los datos originales a los datos filtrados
    def atipicos_electrica(self,columna,logaritmo=True,metodo="LOF",vecinos=3,umbral=0.9):
        self.datos=self.datos[np.where(self.datos[:,columna]>0)] #Se eliminan los datos negativos
        auxiliar=self.datos #Arreglo auxiliar que se inicializa con los datos originales
        auxiliar_anomalias=np.zeros(0) #
        for i in range(self.niveles): #Desplazamiento a través de los niveles
            nivel=i+1
            arreglo=self.datos[np.where(nivel==self.datos[:,0])] #Se extraen los datos del nivel actual
            filtrar=arreglo[:,columna] #Se extraen los datos de la columna a filtrar
            if(logaritmo):
                filtrar=np.array(list(map(math.log10,filtrar))) #Se calcula el logaritmo de cada dato
            n=filtrar.shape[0] #Se obtiene el tamaño del nivel actual
            inmersion=np.zeros((n-1,2)) #Se define el arreglo de inmersión
            k=0 #Contador auxiliar para agregar valores al arreglo inmersión
            for j in range(n-1):
                inmersion[k,0]=filtrar[j]
                inmersion[k,1]=filtrar[j+1]
                k=k+1
            if(metodo=="LOF"):
                modelo=LOF(n_neighbors=vecinos)
                modelo.fit_predict(inmersion)
                factores=modelo.negative_outlier_factor_
                cuantil=np.quantile(factores,umbral) #Cálculo del cuantil
                indices=np.where(factores<=cuantil)
                anomalias=np.zeros(inmersion.shape[0]) #Es 0 para datos típicos
                anomalias[indices]=1 #Es 1 para datos atípicos
            elif(metodo=="IF"):
                modelo=IsFo(max_features=2)
                modelo.fit(inmersion)
                anomalias=modelo.predict(inmersion)
                anomalias[:]=0
                puntaje=modelo.decision_function(inmersion)
                for k in range(puntaje.shape[0]):
                    if(puntaje[k]<-umbral): #Se etiquetan datos anómalos con la puntuación y el umbral
                        anomalias[k]=1
            for k in range(anomalias.shape[0]): #Si hay dos datos atípicos seguidos se elimina el punto en común
                if(k>0 and k<anomalias.shape[0]-1):
                    if(anomalias[k]==1 and anomalias[k-1]==1 and anomalias[k+1]==1):
                        anomalias[k]=1
                    else:
                        anomalias[k]=0
            if(anomalias[-1]==1 and anomalias[-2]==0):
                anomalias=np.append(anomalias,np.array([1]))
            else:
                anomalias=np.append(anomalias,np.array([0]))
            auxiliar_anomalias=np.append(auxiliar_anomalias,anomalias) #Se agrega una columna con las etiquetas de datos atípicos
        auxiliar_anomalias=np.reshape(auxiliar_anomalias,(auxiliar_anomalias.shape[0],1))
        auxiliar=np.append(auxiliar,auxiliar_anomalias,axis=1)
        self.filtrados=auxiliar[np.where(auxiliar[:,-1]==0)] #Se eliminan los datos atípicos
        self.filtrados=np.delete(self.filtrados,-1,1) #Se elimina la columna de etiqueta
    def filtro_ventana_magne(self,columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos):
        x=np.linspace(min(self.datos[:,0]),max(self.datos[:,0]),resolucion_x) #Datos en x
        y=np.linspace(min(self.datos[:,1]),max(self.datos[:,1]),resolucion_y) #Datos en y
        zz=np.zeros((resolucion_y,resolucion_x,0)) #Se crea el arreglo con los datos z de la mall apara cada columna
        for columna in columnas:
            z=griddata((self.datos[:,0],self.datos[:,1]),self.datos[:,columna],(x[None,:],y[:,None]),method='linear')
            z=np.reshape(z,(resolucion_y,resolucion_x,1))
            zz=np.append(zz,z,axis=2)
        fig=plt.figure()
        fig.canvas.mpl_connect('button_press_event',self.evento_mouse) #evento del click en la pantalla
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet) #Solo funciona para la última columna de entrada, esperando que sea gradiente por las unidades (abajo)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.equis=np.array(list(self.equis)) #Se crea el arreglo de coordenadas en x para la ventana
        self.ye=np.array(list(self.ye))  #Se crea el arreglo de coordenadas en y para la ventana
        indices=np.zeros(4) #Arreglo con las 4 coordenadas necesarias, a partir del tercer click se ignora
        for i in range(x.shape[0]): #A partir de aquí se definen los índices para las ventanas
            if(x[i]>=self.equis[0]):
                indices[0]=i
                break
        for i in range(x.shape[0]):
            if(x[i]>=self.equis[1]):
                indices[1]=i
                break
        for i in range(y.shape[0]):
            if(y[i]>=self.ye[0]):
                indices[2]=i
                break
        for i in range(y.shape[0]):
            if(y[i]>=self.ye[1]):
                indices[3]=i
                break
        ventana=zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),-1] #Se aisla la ventana
        ventana=np.reshape(ventana,(ventana.shape[0],ventana.shape[1],1))
        for i in range(len(columnas)-1):
            piv=zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),i]
            piv=np.reshape(piv,(piv.shape[0],piv.shape[1],1))
            ventana=np.append(ventana,piv,axis=2)
        nv=ventana.shape[0] #Tamaño de la ventana
        nvn=ventana.shape[1]
        np2=2
        npn=2
        while np2<nv: #Obtención de la ventana de tamaño potencia de 2 para la transformada wavelet
            np2=np2*2
        while npn<nvn:
            npn=npn*2
        ventanap=np.zeros((np2,npn,len(columnas))) #Nueva ventana
        for i in range(len(columnas)):
            ventanap[0:nv,0:nvn,i]=ventana[:,:,i]
        for j in range(len(columnas)): #Transformada wavelet para cada columna
            coeficientes=pywt.mran(ventanap[:,:,j],wavelet=pywt.Wavelet(ondicula),level=componentes,transform="swtn")
            for i in range(len(coeficientes)): #Atenuación con los pesos
                if(i==0):
                    coeficientes[i]=coeficientes[i]*pesos[i]
                else:
                    coeficientes[i]["ad"]=coeficientes[i]["ad"]*pesos[i]
                    coeficientes[i]["dd"]=coeficientes[i]["dd"]*pesos[i]
                    coeficientes[i]["da"]=coeficientes[i]["da"]*pesos[i]
            ventanap[:,:,j]=pywt.imran(coeficientes) #Transformada wavelet inversa
        for i in range(len(columnas)): #Regreso de la ventana a su posición original dentro de la malla
            ventana[:,:,i]=ventanap[0:nv,0:nvn,i]
            zz[int(indices[3]):int(indices[2]),int(indices[0]):int(indices[1]),i]=ventana[:,:,i]
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.filtrados=np.zeros((0,len(columnas)+2)) #Se guardan los datos filtrados donde las primeras dos columnas son x y y
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                vector=np.array([x[j],y[i]])
                for k in range(len(columnas)):
                    vector=np.append(vector,zz[i,j,k])
                vector=np.reshape(vector,(1,vector.shape[0]))
                self.filtrados=np.append(self.filtrados,vector,axis=0)
    def filtro_malla_magne(self,columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos):
        x=np.linspace(min(self.datos[:,0]),max(self.datos[:,0]),resolucion_x)
        y=np.linspace(min(self.datos[:,1]),max(self.datos[:,1]),resolucion_y)
        zz=np.zeros((resolucion_y,resolucion_x,0))
        for columna in columnas:
            z=griddata((self.datos[:,0],self.datos[:,1]),self.datos[:,columna],(x[None,:],y[:,None]),method='linear')
            z=np.reshape(z,(resolucion_y,resolucion_x,1))
            zz=np.append(zz,z,axis=2)
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        for j in range(len(columnas)): #Se almacenan los índices con nan para igualarlos a cero
            indices_nan=np.zeros((0,2))
            for k in range(zz.shape[0]):
                for kk in range(zz.shape[1]):
                    if(np.isnan(zz[k,kk,j])):
                        vector=np.array([k,kk])
                        vector=np.reshape(vector,(1,2))
                        indices_nan=np.append(indices_nan,vector,axis=0)
                        zz[k,kk,j]=0
            coeficientes=pywt.mran(zz[:,:,j],wavelet=pywt.Wavelet(ondicula),level=componentes,transform="swtn")
            for i in range(len(coeficientes)):
                if(i==0):
                    coeficientes[i]=coeficientes[i]*pesos[i]
                else:
                    coeficientes[i]["ad"]=coeficientes[i]["ad"]*pesos[i]
                    coeficientes[i]["dd"]=coeficientes[i]["dd"]*pesos[i]
                    coeficientes[i]["da"]=coeficientes[i]["da"]*pesos[i]
            zz[:,:,j]=pywt.imran(coeficientes)
            for i in range(indices_nan.shape[0]):
                zz[int(indices_nan[i,0]),int(indices_nan[i,1]),j]=np.nan #Se regresa nan a las coordenadas que lo tenían originalmente
        plt.contourf(x,y,zz[:,:,-1],16,cmap=plt.cm.jet)
        plt.colorbar().set_label("nT/m")
        plt.title("Base de datos magnetometría columna "+str(columnas[-1]))
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.show()
        plt.clf()
        self.filtrados=np.zeros((0,len(columnas)+2))
        for i in range(y.shape[0]):
            for j in range(x.shape[0]):
                vector=np.array([x[j],y[i]])
                for k in range(len(columnas)):
                    vector=np.append(vector,zz[i,j,k])
                vector=np.reshape(vector,(1,vector.shape[0]))
                self.filtrados=np.append(self.filtrados,vector,axis=0)
    def ondicula_magnetometria(self,columnas,resolucion_x=128,resolucion_y=128,malla=True,ventana=False,ondicula="db4",componentes=3,pesos=[1,0.3,0.6,1],pesos2=[1,0.3,0.6,1]):
        if(ventana and not malla):
            self.filtro_ventana_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
        elif(malla and not ventana):
            self.filtro_malla_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
        elif(malla and ventana):
            self.filtro_ventana_magne(columnas,resolucion_x,resolucion_y,ondicula,componentes,pesos)
            self.datos=self.filtrados
            n=len(columnas)
            c=[]
            for i in range(n):
                c.append(i+2)
            self.filtro_malla_magne(c,resolucion_x,resolucion_y,ondicula,componentes,pesos2)
    def guardar(self,nombre):
        formato=[]
#        if(self.tipo=="magnetometría"):
#            filtrados=self.filtrados[:,0:2]
#            for i in range(self.filtrados.shape[1]):
#                formato.append("%0.8f")
#                if(i>1):
#                    x=self.datos[:,0]
#                    y=self.datos[:,1]
#                    f=griddata((self.filtrados[:,0],self.filtrados[:,1]),self.filtrados[:,i],(x[None,:],y[:,None]),method="nearest")
#                    f=np.reshape(f,(x.shape[0],1))
#                    filtrados=np.hstack((filtrados,f))
#            self.filtrados=filtrados
#        elif(self.tipo=="eléctrica"):
        for i in range(self.filtrados.shape[1]):
            formato.append("%0.8f")
        np.savetxt(nombre,self.filtrados,delimiter=",",fmt=formato)
