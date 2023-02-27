from preprocessing import *

###
# ERT2D semi-automatic modelling
###

##
# Malla.vtk is the grid with the base model, the "Malla_inicio.vtk" and "Malla_fin.vtk" are the first and last sections of the "Malla.vtk" excluding the resistivity data
# intervalos is the variation range of the random generated resistivities to build the model, according to the base model
# secuencias are the files that contains the the automatic sequence archives for the model generation, with the format required by the ResIPy API
# modelos is the number of models to generate
# electrodos are the number of electrodes per automatic sequence
# dx is the electrode spacing
##

intervalos=[[0.01,10],[10,100],[100,1000]]
secuencias=["WS.csv","DD.csv"]
modelos=10
electrodos=24
dx=5.0
vecinos=3
modelacion=modelar_electrica_2D("Malla.vtk","Malla_inicio.vtk","Malla_fin.vtk")
modelacion.modelos(intervalos,secuencias,modelos,electrodos,dx)
modelacion.agrupar_secuencias(vecinos)
modelacion.agrupar_modelo()

###
# Database preparation for the training and clustering steps
###

##
# carpeta is the path of the databases
# datos are the files names that contains the databases to preprocess
##

carpeta_magnetometria="Magnetometría/"
carpeta_electrica="Eléctrica/"
datos_magnetometria=["BOT.csv","TOP.csv","GRAD.csv","BOT_RP.csv","TOP_RP.csv","BOT_SA.csv","TOP_SA.csv","BOT_GH.csv","TOP_GH.csv","BOT_TD.csv","TOP_TD.csv","BOT_GHTD.csv","TOP_GHTD.csv"]
datos_electrica=["WS","DD"]
vecinos_magnetometria=4
vecinos_electrica=3
preparados_magnetometria=preparar(carpeta_magnetometria)
preparados_magnetometria.crear_base_magnetometria(datos_magnetometria)
preparados_magnetometria.vecindad(vecinos_magnetometria)
preparados_magnetometria.transformar(metodo="ecualizar",bins="auto")
preparados_magnetometria.guardar("Malla_mag.csv")
preparados_electrica=preparar(carpeta_electrica)
preparados_electrica.crear_base_electrica(datos_electrica)
preparados_electrica.vecindad(vecinos_electrica)
preparados_electrica.transformar(metodo="logaritmica")
preparados_electrica.guardar("TRE.csv")
