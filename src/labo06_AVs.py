import numpy as np
import sys
sys.path.append(".") 
from labo00_auxiliares import *
from labo01_errores_igualdad import *
from labo03_normas import norma
from labo05_QR import *
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





def diagRH(A, tol =1e-8 ,K=1000):
    """
    A: una matriz simetrica de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S. T
    Si la matriz A no es simetrica, debe retornar None.
    """
    if not cuadrada(A) or not (esS:=esSimetrica(A, tol)):
        # si lo dejo haciendo cuentas grandes no quiero perderlo todo pq estaba mal la toleranica
        print("ERROR: se rompe cuadrada o simetria con", A.shape, "o alguna tolerancia que no quiero gastar tiempo en calcular" )
        # return None
    n = len(A)
    if n % 20 == 0:
        print(f"diagonalizando {n}-esima sumbatriz a las {datetime.now().time()}")


    v,aVal,_ = metpot2k(A,tol,K)
    e1_menos_v = identidad(n)[0] - v
    # Hv = houseHolder( e1_menos_v / norma(e1_menos_v, 2) )
    v_para_householder = e1_menos_v / norma(e1_menos_v, 2)
    if n == 2:
        S_matriz_avecs = houseHolder(v_para_householder)
        D_matriz_avals = matmulHouseHolderIzquierda(v_para_householder, matmulHouseHolderDerecha(A, v_para_householder))
    else:
        B = matmulHouseHolderIzquierda(v_para_householder, matmulHouseHolderDerecha(A, v_para_householder))
        ASombrero = B[1:,1:]
        SSombrero,DSombero  = diagRH(ASombrero, tol, K)
        D_matriz_avals = expandirDiagonalPrincipalDesdeArriba(DSombero, aVal)  
        S_matriz_avecs = matmulHouseHolderIzquierda(v_para_householder,
            expandirDiagonalPrincipalDesdeArriba(SSombrero, 1))      

    if n % 20 == 0:
        print(f"listo diagonalizando {n}-esima sumbatriz a las {datetime.now().time()}")


    return S_matriz_avecs, D_matriz_avals
