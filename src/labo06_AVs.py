import numpy as np
import sys
sys.path.append(".") 
from labo00_auxiliares import *
from labo01_errores_igualdad import *
from labo03_normas import norma
from labo05_QR import houseHolder
from datetime import datetime


def aplicarPotenciaUnaVez(A,v):
    vSombrero = calcularAx(A, v)
    norm = norma(vSombrero,2)
    if norm == 0:
        return 0
    vSombrero = vSombrero / norm
    return vSombrero

def aplicarPotenciaDosVeces(A,v):
    
    vSombrero = aplicarPotenciaUnaVez(A,aplicarPotenciaUnaVez(A,v))

    e = prodint(vSombrero, v)
    return vSombrero, e

def metpot2k(A, tol =10**(-15) ,K=1000):
    """
    A: una matriz de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    Retorna vector v, autovalor lambda y numero de iteracion realizadas k.
    """
    n = len(A)
    if not cuadrada(A):
        return None
    v = np.random.randn(n)
    vSombrero, e = aplicarPotenciaDosVeces(A, v)
    for k in range(int(K)):
        if abs(abs(e)-1)<=tol:
            break
        v = vSombrero
        vSombrero, e = aplicarPotenciaDosVeces(A, v)

    aVal = prodint(vSombrero, calcularAx(A, vSombrero))
    err = e-1

    return vSombrero, aVal, k


#def reflectorHouseholder(v): # pq es correcto esto? parece raro
#    n = len(v)
#    idn = identidad(n)
#    uut = matmul([idn[0]-v], [idn[0]-v])
#   return idn - 2 * (uut / norma(uut,2))



def diagRH(A, tol =10**(-15) ,K=1000):
    """
    A: una matriz simetrica de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S. T
    Si la matriz A no es simetrica, debe retornar None.
    """
    if not cuadrada(A) or not esSimetrica(A):
        return None
    n = len(A)
    if n % 100 == 0:
        print(f"diagonalizando {n}-esima sumbatriz a las {datetime.now().time()}")


    v,aVal,_ = metpot2k(A,tol,K)
    e1_menos_v = identidad(n)[0] - v
    Hv = houseHolder( e1_menos_v / norma(e1_menos_v, 2))
    if n == 2:
        S_matriz_avecs = Hv
        D_matriz_avals = matmul(Hv, matmul(A, traspuesta(Hv)))
    else:
        B = matmul(Hv, matmul(A, traspuesta(Hv)))
        ASombrero = B[1:,1:]
        SSombrero,DSombero  = diagRH(ASombrero, tol, K)
        D_matriz_avals = expandirDiagonalPrincipalDesdeArriba(DSombero, aVal)  
        S_matriz_avecs = matmul(Hv,
            expandirDiagonalPrincipalDesdeArriba(SSombrero, 1))      


    return S_matriz_avecs, D_matriz_avals
