import numpy as np
from labo00_auxiliares import *
from labo03_normas import norma
from labo01_errores_igualdad import epsilon, feq


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
    for k in range(K):
        if abs(abs(e)-1)<=tol:
            break
        v = vSombrero
        vSombrero, e = aplicarPotenciaDosVeces(A, v)

    aVal = prodint(vSombrero, calcularAx(A, vSombrero))
    err = e-1

    return vSombrero, aVal, k, err


def reflectorHouseholder(v):
    n = len(v)
    idn = np.identity(n)
    uut = matmul(idn[0]-v, traspuesta(idn[0]-v))
    return idn - 2 * (uut / norma(uut,2))

def expandirDiagonalPrincipalDesdeArriba(D, zerozero):
    D = np.insert(D, 0, len(D),0)
    D = np.insert(D, 0, len(D),1)
    D[0][0] = zerozero
    return D


def diagRH(A, tol =10**(-15) ,K=1000):
    """
    A: una matriz simetrica de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S. T
    Si la matriz A no es simetrica, debe retornar None.
    """
    if not cuadrada(A):
        return None
    n = len(A)

    v,aVal,_,_ = metpot2k(A,tol,K)
    Hv = reflectorHouseholder(v)

    if n == 2:
        S_matriz_avecs = Hv
        D_matriz_avals = matmul(Hv, matmul(A, traspuesta(Hv)))
    else:
        B = matmul(Hv, matmul(A, traspuesta(Hv)))
        ASombrero = B[1:,1:]
        SSombrero, DSombero = diagRH(ASombrero, tol, K)
        D = expandirDiagonalPrincipalDesdeArriba(DSombero, aVal)  
        S = matmul(Hv,
            expandirDiagonalPrincipalDesdeArriba(SSombrero, 1))      


    pass
