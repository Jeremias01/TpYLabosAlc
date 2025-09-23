### Funciones L05-QR
import numpy as np
from labo00_auxiliares import *
from labo03_normas import norma
from labo01_errores_igualdad import epsilon, feq

"""
A una matriz de n x n 
tol la tolerancia con la que se filtran elementos nulos en R
retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
Si la matriz A no es de n x n, debe retornar None
"""

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    if not cuadrada(A):
        return None
    n=len(A)
    contador=0
    Q=np.zeros((n,n))
    R=np.zeros((n,n))
    AColumnas = traspuesta(A).astype(np.float64)

    R[0][0]=norma(AColumnas[0],2)


    if feq(R[0][0] , 0):
        Q[0] = np.zeros((n))
    else:
        Q[0]=AColumnas[0]/R[0][0]

    for j in range(1,n):
        Qj=AColumnas[j]
        for k in range(0,j):
            R[k][j]=prodint(Q[k],Qj)
            Qj+=-R[k][j]*Q[k]
        R[j][j]=norma(Qj,2)

        if feq(R[j][j] , 0):
            Q[j] = np.zeros((n,))
        else:
            Q[j]=Qj*(1/R[j][j])
    if(retorna_nops):
        return traspuesta(Q),R,contador
    else: return traspuesta(Q),R 


#A=[[12,-51,4],[12,-52,4],[-4,24,-41]]
A=[[1,1,0],[1,1,0],[0,0,1]]
print(QR_con_GS(A))



def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """