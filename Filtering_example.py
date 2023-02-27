from filtering import *

###
# ERT2D filtering through the DWT1D
###

##
# archivo.csv is a comma delimited file with nxm shape with a one-row header
##

columna=7
filtro=filtrado("archivo.csv","eléctrica")
filtro.ondicula_electrica(columna,ondicula="db2",componentes=5,recursion=1)

#
# Next, the system waits for the input weights for each detail component
#

filtro.guardar("archivo_filtrado.csv")

###
# ERT2D anomalous data removal
# Can be applied after the DWT1D filter
###

##
# The file's first column must be the acquisition point depth level. Necessarily the data must be in asceding order according to the depth level
##

filtro=filtrado("archivo.csv","eléctrica")
filtro.atipicos_electrica(columna,logaritmo=True,metodo="LOF",vecinos=3,umbral=0.9)
filtro.guardar("archivo_filtrado.csv")

###
# Magnetometry grid data filtering through DWT2D
###

columnas=[2,3,4]
filtro=filtrado("archivo.csv","magnetometría")
filtro.ondicula_magnetometria(columnas,resolucion_x=32,resolucion_y=32,malla=True,ventana=True,ondicula="db4",componentes=3,pesos=[0,0.9,0.4,0],pesos2=[1,0.3,0.6,1])

##
# Next, the user clicks on the desired window's upper right and bottom left corners
##

filtro.guardar("archivo_filtrado.csv")
