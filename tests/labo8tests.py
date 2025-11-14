# Tests L08
import numpy as np
import sys
sys.path.append("./src")
from labo08_SVD import *
    
# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    print(f"S nuestra:\n {hS} S NP:\n {np.round(nS, 2)}")
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))
    # print(f"A ex:\n {A} A nuestra:\n {hU @ np.diag(hS) @ traspuesta(hV)}")

    assert np.allclose(A, hU @ np.diag(nS) @ traspuesta(hV)), 'multiplicacion no da lo correcto '

    print(f"V nuestra:\n {np.round(traspuesta(hV), 2)} V NP:\n {np.round(hV, 2)}")
    print(f"V@VT nuestra:\n {hV.T @ hV}")
    # estas 2 lineas etan gpteadas
    print(np.all(np.isclose(np.diag(hV.T @ hV), np.ones(hV.shape[1]), atol=1e-5)))
    assert np.all(np.isclose(np.diag(hV.T @ hV), np.ones(hV.shape[1]), atol=1e-5))
    assert np.all(np.isclose(np.diag(hV @ hV.T), np.ones(hV.shape[0]), atol=1e-5))

    print(f"U nuestra:\n {np.round(hU, 2)} U NP:\n {np.round(hU, 2)}")
    print(f"U@UT nuestra:\n {np.round(hU.T @ hU, 15)}")
    print(np.abs(np.abs(np.diag(hU.T @ hU))-1).astype(str), 10**r*tol, np.abs(np.abs(np.diag(hU @ hU.T))-1)<10**r*tol)
    assert np.all(np.isclose(np.abs(np.abs(np.diag(hU.T @ hU))-1), np.zeros(len(np.diag(hU.T @ hU))))), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.isclose(np.abs(np.abs(np.diag(hU @ hU.T))-1), np.zeros(len(np.diag(hU @ hU.T))))), 'Revisar calculo de hat U en ' + str((m,n))



def run():

    for m in [2,5,10,20]:
        for n in [2,5,10,20]:
            for _ in range(10):
                A = genera_matriz_para_test(m,n)
                test_svd_reducida_mn(A)


    # Matrices con nucleo

    m = 12
    for tam_nucleo in [2,4,6]:
        for _ in range(10):
            A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
            test_svd_reducida_mn(A)

    # Tamaños de las reducidas
    A = np.random.random((8,6))
    for k in [1,3,5]:
        hU,hS,hV = svd_reducida(A,k=k)
        assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
        assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
        assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
        assert len(hS) == k, 'Tamaño de hS incorrecto'

run()