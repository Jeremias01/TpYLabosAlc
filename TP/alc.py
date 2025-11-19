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
from datetime import datetime
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
    # if os.getcwd()[-3:] != "/TP":
    #     prefix = "TP"

    XvCats = np.load(f'{os.getcwd()}{prefix}/{carpeta}/val/cats/efficientnet_b3_embeddings.npy')
    XtCats = np.load(f'{os.getcwd()}{prefix}/{carpeta}/train/cats/efficientnet_b3_embeddings.npy')
    XvDogs = np.load(f'{os.getcwd()}{prefix}/{carpeta}/val/dogs/efficientnet_b3_embeddings.npy')
    XtDogs = np.load(f'{os.getcwd()}{prefix}/{carpeta}/train/dogs/efficientnet_b3_embeddings.npy')

    Xv = np.concatenate((XvDogs, XvCats), 1)    
    Xt = np.concatenate((XtDogs, XtCats), 1)    

    YvCats = np.concatenate(tuple([[1],[0]] for x in range(XvCats.shape[1])),1)
    YvDogs = np.concatenate(tuple([[0],[1]] for x in range(XvDogs.shape[1])),1)
    YtCats = np.concatenate(tuple([[1],[0]] for x in range(XtCats.shape[1])),1)
    YtDogs = np.concatenate(tuple([[0],[1]] for x in range(XtDogs.shape[1])),1)

    Yv = np.concatenate((YvDogs, YvCats), 1)    
    Yt = np.concatenate((YtDogs, YtCats), 1)    


    return Xt, Yt, Xv, Yv


def cargarDatasetReducido(carpeta, filas, columnas):
    """
    columnas debe ser mayor a 2*filas para que la matriz que devuelva sea compatible con los otros metodos
    para probar con un subconjunto del dataset. filas es cuantas filas de cada embedding, y columnas cuantos embeddings de perros y de gatos
    devuelve matriz de filas x 2*columnas.
    """
    Xt, Yt, Xv, Yv = cargarDataset(carpeta)
    return (np.concatenate((Xt[:filas,:columnas], Xt[:filas,-columnas:]),1),
            np.concatenate((Yt[:,:columnas] , Yt[:,-columnas:]),1),
            np.concatenate((Xv[:filas,:columnas] , Xv[:filas,-columnas:]),1),
            np.concatenate((Yv[:,:columnas] , Yv[:,-columnas:]),1),
        )

# Esta funcion obtiene la matriz L (Lower) utilizada al factorizar por Cholesky de la forma A = L*L^t
# Acepta solo matrices A diagonales, que es nuestro caso de uso
def cholesky(A):
    L = np.zeros((len(A),len(A[0])))
 
    for columna in range((len(A))):
        if columna % 100 == 0:
            print(f"choleskizando {columna}-esima columna a las {datetime.now().time()}")
        
        #suma = np.sum(np.power(L[columna],2))
        suma = 0
        for i in range(columna):
            suma += (L[columna][i])**2
        
        diagonal = np.sqrt(A[columna][columna]- suma)
        L[columna][columna] = diagonal

        for fila in range(columna+1, len(A)):
            suma = np.sum(L[fila][:columna] * L[columna][:columna])
            
            L[fila][columna] = (A[fila][columna] - suma) / L[columna][columna]

    return L

def pinvEcuacionesNormales(X,L, Y):
    """
    Resuelve el cálculo de los pesos utilizando las ecuaciones normales para
    la resolución de la psetargetsudo-inversa usando el algoritmo 1 y descomposición cholesky. 
    X: la matriz original
    L: la matriz de Cholesky de XX^t 
    Y: la matriz de  de entrenamiento.
    retorna cálculo de los pesos W
    """
    # el enunciado pide usar  V X X^t    = X^t.
    #                        (V X X^t)^t = X
    #                         X X^t V^t = X
    #                         L L^t V^t = X
    
    Vt = res_LU_mat(L, traspuesta(L), X)
    V = traspuesta(Vt)
    W = matmul(Y,V)
    return W


def pinvSVD(U, S, V, Y, tol=1e-8):
    """ 
    Obtiene los pesos utilizando la Descomposición
    en Valores Singulares para la resolución de la pseudo-inversa usando el algoritmo 2. 
    U: Matriz de autovectores por izquierda de SVD reducida
    S: Matriz Sigma de valores singulares 
    V: Matriz de autovectores por derecha de SVD reducida
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    # Queremos un vector S_inv con valores singulares > 0

    k = min(S.shape[0],S.shape[1])
    S_inv = np.zeros(k)
    i = 0
    while i < k and S[i][i] > tol:
        S_inv[i] = S[i][i] ** -1
        i += 1
    S_inv = S_inv[:i]


    # Como tenemos S_inv vector que representa una matriz diagonal, podemos hacer mas rápida la multiplicación
    VS_inv = np.zeros(V.shape)
    for i in range(V.shape[1]):
        VS_inv[:, i] = V[:, i] * S_inv[i]
    
    A_pseudoinv = matmul(VS_inv, traspuesta(U))
    #assert esPseudoInversa(U @ np.diag(S) @ traspuesta(V), A_pseudoinv, 1e-8)
    
    W = matmul(Y,A_pseudoinv)

    return W
    

def pinvQR(Q,R,Y):
    VT = res_tri_mat(R, traspuesta(Q), False)
    print("listo resolviendo sistema")
    V = traspuesta(VT)
    #assert esPseudoInversa(traspuesta(Q @ R), V)
    print("listo trasponiendo")
    W = matmul(Y,V)
    
    print("Calculando W")
    return W


def pinvHouseHolder(Q, R, Y):
    """
    Usando factorizacion QR de X^T,
    Q: Matriz ortonormal de QR, calculada con HouseHolder
    R: Matriz triangular superior de QR, calculada con HouseHolder
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    return pinvQR(Q,R,Y)


def pinvGramSchmidt(Q, R, Y):
    """
    Usando factorizacion QR de X^T,
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
    pasa_condiciones &= matricesIguales( matmul(XpX, X)  ,X,  tol )
    pasa_condiciones &= matricesIguales( matmul(pXX, pX) ,  pX , tol)
    pasa_condiciones &= esSimetrica( XpX , tol)
    pasa_condiciones &= esSimetrica( pXX , tol)
    

    return pasa_condiciones




