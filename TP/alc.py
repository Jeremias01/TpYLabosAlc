import numpy as np
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
from labo07_markov import *
from labo08_SVD import *
### TODOS ESTOS IMPORTS SE VAN A TENER QUE REEMPLAZAR CON EL CODIGO ENTERO EN LA ENTREGA FINAL TODO

def cargarDataset(carpeta):
    """
    carpeta: un string con dirección a la carpeta con datos de entrenamiento
    retorna matrices Xt, Yt, Xv, Yv. 
    Las matrices (Xt, Yt) corresponden a los ejemplos de entrenamiento conteniendo
    los embeddings de los gatos y perros juntos. 
    Cada columna de la matriz corresponde a un embedding. 
    La matriz Yt debe generarse a partir de la lectura de los archivos de entrenamiento.
    Cada columna de Yt tiene 2 elementos valiendo 0 o 1 dependiendo de la clase 
    a la que pertenece el embedding. 
    Por ejemplo un yi de un gato debería ser yi = [1, 0]T , y otro de un perro: yj = [0, 1]T .
    """
    
    pass

def pinvEcuacionesNormales(L, Y):
    """
    Resuelve el cálculo de los pesos utilizando las ecuaciones normales para
    la resolución de la pseudo-inversa usando el algoritmo 1 y descomposición cholesky. 
    L: la matriz de Cholesky 
    Y: la matriz de targets de entrenamiento.
    retorna cálculo de los pesos W
    """
    
    pass

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
    
    pass

def pinvHouseHolder(Q, R, Y):
    """
    Q: Matriz ortonormal de QR, calculada con HouseHolder
    R: Matriz triangular superior de QR, calculada con HouseHolder
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    pass

def pinvGramSchmidt(Q, R, Y):
    """
    Q: Matriz ortonormal de QR, calculada con Gram-Schmidt
    R: Matriz triangular superior de QR, calculada con Gram-Schmidt
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    
    pass

def esPseudoInverda(X, pX, tol=1e-8):
    """
    X: matrix
    pX: matrix
    retorna True si y solo si las X y xP verifican las condiciones de Moore-Penrose 
    """
    pass