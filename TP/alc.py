import numpy as np
import sys
import os
sys.path.append(".") 
sys.path.append("../src") 
sys.path.append("./src") 
from labo00_auxiliares import *
from labo01_errores_igualdad import *
from labo02_TLs_basicas import *
from labo03_normas import *
from labo04_LU import *
from labo05_QR import *
from labo06_AVs import *
from labo07_markov import *
from labo08_SVD import *
### TODOS ESTOS IMPORTS SE VAN A TENER QUE REEMPLAZAR CON EL CODIGO ENTERO EN LA ENTREGA FINAL TODO



def cargarDataset(carpeta):
    """
    carpeta: un string con dirección a la carpeta con datos de entrenamiento
    retorna matrices Xt, Yt, Xv, Yv. 
    Las matrices (Xt, Yt) corresponden a los ejemplos de entrenamiento conteniendo los embeddings de
    los gatos y perros juntos. Cada columna de la matriz corresponde a un embedding. La matriz Yt
    debe generarse a partir de la lectura de los archivos de entrenamiento. Cada columna de Yt tiene
    2 elementos valiendo 0 o 1 dependiendo de la clase a la que pertenece el embedding. 
    Por ejemplo un yi de un gato debería ser yi = [1, 0]T , y otro de un perro: yj = [0, 1]T .
    """
    # Esta parte de PREFIX se tiene que borrar cuando ya este todo funcionando para la entrega final TODO 
    
    prefix = ""
    if os.getcwd()[-3:] != "/TP":
        prefix = "TP"

    XvCats = np.load(f'{os.getcwd()}{prefix}/{carpeta}/val/cats/efficientnet_b3_embeddings.npy')
    XtCats = np.load(f'{os.getcwd()}{prefix}/{carpeta}/train/cats/efficientnet_b3_embeddings.npy')
    XvDogs = np.load(f'{os.getcwd()}{prefix}/{carpeta}/val/dogs/efficientnet_b3_embeddings.npy')
    XtDogs = np.load(f'{os.getcwd()}{prefix}/{carpeta}/train/dogs/efficientnet_b3_embeddings.npy')

    Xv = np.concatenate((XvDogs, XvCats), 1)    
    Xt = np.concatenate((XtDogs, XtCats), 1)    

    #TODO no se cual es 1 0 y cual es 0 1
    YvCats = np.concatenate(tuple([[1],[0]] for x in range(XvCats.shape[1])),1)
    YvDogs = np.concatenate(tuple([[0],[1]] for x in range(XvDogs.shape[1])),1)
    YtCats = np.concatenate(tuple([[1],[0]] for x in range(XtCats.shape[1])),1)
    YtDogs = np.concatenate(tuple([[0],[1]] for x in range(XtDogs.shape[1])),1)

    Yv = np.concatenate((YvDogs, YvCats), 1)    
    Yt = np.concatenate((YtDogs, YtCats), 1)    


    return Xt, Yt, Xv, Yv


def cargarDatasetReducido(carpeta, filas, columnas):
    Xt, Yt, Xv, Yv = cargarDataset(carpeta)
    return (np.concatenate((Xt[:filas,:columnas], Xt[:filas,-columnas:]),1),
            np.concatenate((Yt[:,:columnas] , Yt[:,-columnas:]),1),
            np.concatenate((Xv[:filas,:columnas] , Xv[:filas,-columnas:]),1),
            np.concatenate((Yv[:,:columnas] , Yv[:,-columnas:]),1),
        )

# Esta funcion obtiene la matriz L (Lower) utilizada al factorizar por Cholesky de la forma A = L*L^t
def cholesky(A):
    L = np.zeros((len(A),len(A[0])))
 
    for columna in range((len(A))):
        suma = 0
        for i in range(columna):
            suma += (L[columna][i])**2
        diagonal = np.sqrt(A[columna][columna]- suma)
        L[columna][columna] = diagonal

        for fila in range(columna+1, len(A)):
            suma = 0
            for i in range(0,columna):
                suma += L[fila][i] * L[columna][i]
            L[fila][columna] = (A[fila][columna] - suma) / L[columna][columna]

    return L

# Esta funcion la utilizamos para obtener Z, nos interesa obtener Z para resolver la ecuacion L*Z = X^t

def sustitucion_adelante(L, B):
    p = L.shape[0]
    Z = np.zeros((p, B.shape[1]))   

    for i in range(p):               
        for c in range(B.shape[1]):   
            suma = 0.0
            for k in range(i):       
                suma += L[i][k] * Z[k][c]
            Z[i][c] = (B[i][c] - suma) / L[i][i]

    return Z

# Esta funcion nos permite obtener X^+ utilizando el Z obtenido por la funcion anterior, esto se obtiene a partir de la ecuacion L^t * X^+ = Z
def sustitucion_atras(U,Z):
    p = U.shape[0]
    V = np.zeros((p, Z.shape[1])) 

    for i in range(p - 1, -1, -1):       
        for c in range(Z.shape[1]):     
            suma = 0.0
            for k in range(i + 1, p):    
                suma += U[i][k] * V[k][c]
            V[i][c] = (Z[i][c] - suma) / U[i][i]

    return V


#TODO che no tenemos permitido usar .T ni @
#Esta funcion toma L (cholesky), Y y calcula W usando dos sustituciones: una para llegar a Z y otra para llegar a la pseudo–inversa X^+.        
def pinvEcuacionesNormales(L, Y):
    """
    Resuelve el cálculo de los pesos utilizando las ecuaciones normales para
    la resolución de la psetargetsudo-inversa usando el algoritmo 1 y descomposición cholesky. 
    L: la matriz de Cholesky 
    Y: la matriz de  de entrenamiento.
    retorna cálculo de los pesos W
    """
    Z = sustitucion_adelante(L, Xt)
    Xp = sustitucion_atras(L.T, Z)
    W = Y @ Xp
    return W
   

def pinvSVD(U, S, V, Y):
    """ 
    Obtiene los pesos utilizando la Descomposición
    en Valores Singulares para la resolución de la pseudo-inversa usando el algoritmo 2. 
    U: Matriz de autovectores por izquierda de SVD
    S: Matriz Sigma de valores singulares
    V: Matriz de autovectores por derecha de SVD
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    S_ALaMenosUno = np.zeros((len(S),len(S[0])))
    for i in len(S):
        if S[i][i] >0:               # si uno es cero, los que le siguen tambien
            S_ALaMenosUno[i][i] = (S[i][i])**(-1)
    SigmaMas = traspuesta(S_ALaMenosUno[:,len(S)])
    V1 = V[:,len(S)]
    W = matmul(Y, matmul((V1, SigmaMas),traspuesta(U)))

    return W
    

def pinvQR(Q,R,Y):
    VT = res_tri_mat(R, traspuesta(Q), False)
    print("listo resolviendo sistema")
    V = traspuesta(VT)
    print("listo trasponiendo")
    W = matmul(Y,V)
    print("Calculando W")
    return W


def pinvHouseHolder(Q, R, Y):
    """
    Q: Matriz ortonormal de QR, calculada con HouseHolder
    R: Matriz triangular superior de QR, calculada con HouseHolder
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    return pinvQR(Q,R,Y)


def pinvGramSchmidt(Q, R, Y):
    """
    Q: Matriz ortonormal de QR, calculada con Gram-Schmidt
    R: Matriz triangular superior de QR, calculada con Gram-Schmidt
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    
    return pinvQR(Q,R,Y)

def esPseudoInversa(X, pX, tol=1e-8):
    """
    X: matrix
    pX: matrix
    retorna True si y solo si las X y xP verifican las condiciones de Moore-Penrose 
    """
    XpX = matmul(X, pX)
    pXX = matmul(pX, X)
    
    pasa_condiciones = True
    pasa_condiciones &= matricesIguales( matmul(XpX, X)  ,  X )
    pasa_condiciones &= matricesIguales( matmul(pXX, pX) ,  pX )
    pasa_condiciones &= esSimetrica( XpX )
    pasa_condiciones &= esSimetrica( pXX )
    
    return pasa_condiciones




