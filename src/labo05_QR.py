### Funciones L05-QR
import numpy as np
from labo00_auxiliares import *
from labo03_normas import norma
from labo01_errores_igualdad import epsilon, feq
from datetime import datetime

"""
A una matriz de n x n 
tol la tolerancia con la que se filtran elementos nulos en R
retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
Si la matriz A no es de n x n, debe retornar None
"""

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    #if not A.shape[1] >= A.shape[0]: # el apunte del TP pide mas filas que columnas. la toerica decia mas columnas que filas. esta es la que funciona
    #    return None
    m = len(A)
    n = len(A[0])
    

    contador=0
    QT=np.zeros((n,m)) 
    R=np.zeros((n,n)) # R=np.zeros((k,n)) si no queres los 0s de mas
    AColumnas = traspuesta(A).astype(np.float64)

    R[0][0]=norma(AColumnas[0],2)

    # definimos primer columna de Q como la primera de A normalizada
    if feq(R[0][0] , 0):
        QT[0] = np.zeros((n))
    else:
        QT[0]=AColumnas[0]/R[0][0]

    # por cada columna que queda, 
    for j in range(1,n):
        if j % 100 == 0:
            print(f"ortonormalizando {j}-esimo vector de {n} a las {datetime.now().time()}")
        Qj=AColumnas[j]
        for k in range(0,j):
            R[k][j]=prodint(QT[k],Qj)
            Qj+=-R[k][j]*QT[k]
        R[j][j]=norma(Qj,2)

        if feq(R[j][j] , 0):
            QT[j] = np.zeros((n,))
        else:
            QT[j]=Qj*(1/R[j][j])
    
    
    if retorna_nops :
        return traspuesta(QT),R,contador
    else: 
        return traspuesta(QT),R 


#A=[[12,-51,4],[12,-52,4],[-4,24,-41]]
#A=[[1,1,0],[1,1,0],[0,0,1]]
#print(QR_con_GS(A))


def houseHolder(u):
    # precondición: u tal que ||u||_2 = 1
    mMenosK = len(u) 
    return identidad(mMenosK) - 2 * matCol(u)@matFila(u)           # cambio los matmul por @ para que no sea tan lento



def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    if ( m := len(A) ) == 0 or m < ( n := len(A[0])):
        return None
    
    R = A.astype(np.float64)
    Q = identidad(m)
    for k in range(0,n):
        if k % 10 == 0:
            print(f"householderizando {k}-esima sumbatriz de {n} a las {datetime.now().time()}")

        x = R[k:, k]
        alpha = - sign(x[0]) * norma(x, 2)     # doble negacion redundante??
        u = x - alpha * identidad(m-k)[0]
        if (unorma := norma(u, 2)) > tol:
            u = u / unorma
            Hk =  houseHolder(u)
            for iter in range(k): 
                Hk = expandirDiagonalPrincipalDesdeArriba(Hk, 1)
            R = Hk@R
            Q = Q@traspuesta(Hk)       # cambio los matmul por @ para que no sea tan lento

    #borramos filas y columnas redundantes antes de devolver
    return Q[:,:n],R[:n,:]



def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    match metodo:
        case 'RH':
            return QR_con_HH(A,tol=1e-12)
        case 'GS':
            # el enunciado dice que con tercer parametro opcional el numero de operaciones, pero no
            # es argumeto a calculaQR, no tiene sentido (?)
            return QR_con_GS(A,tol=1e-12)
        