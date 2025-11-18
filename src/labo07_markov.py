import sys

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

import numpy as np

def transiciones_al_azar_continuas(n):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el
    intervalo [0, 1]
    """
    n = np.random.rand(n,n)
    for i in range(n):
        n[:, i] = n[:, i] / norma(n[:, i],1)
    return n




def transiciones_al_azar_uniforme(n, thres):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    thres: probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. El elemento i, j es distinto de
    cero si el número generado al azar para i, j es menor o igual a thres. Todos los
    elementos de la columna $j$ son iguales (a 1 sobre el número de elementos distintos
    de cero en la columna).
    """
    n = np.random.rand(n,n)
    for i in range(n):
        n[:, i] = n[:, i] *  (n[:, i>thres].astype(np.float64)) # dudoso
        n[:, i] = n[:, i] / norma(n[:, i],1)
    return n



def nucleo(A, tol=1e-15):
    """
    A: una matriz de m x n
    tol: la tolerancia para asumir que un vector está en el núcleo.
    Calcula el núcleo de la matriz A diagonalizando la matriz traspuesta (A) * A (* la
    multiplicación matricial), usando el medodo diagRH. El núcleo corresponde a los
    autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestión, como una matriz de n x k, con k el numero de
    autovectores en el núcleo.
    """
    
    raise NotImplementedError("Implementar")



def crear_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con índices i, lista con índices
    j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a
    traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores
    distintos de cero. Retorna una lista con:
    - Diccionario { (i, j): A_ij } que representa los elementos no nulos de la matriz A. Los
    elementos con modulo menor a tol deben descartarse por default.
    - Tupla (m_filas, n_columnas) que permita conocer las dimensiones de la matriz.
    """
    raise NotImplementedError("Implementar")



def multiplicar_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crear_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """
    raise NotImplementedError("Implementar")
