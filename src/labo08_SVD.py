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

### TODOS ESTOS IMPORTS SE VAN A TENER QUE REEMPLAZAR CON EL CODIGO ENTERO EN LA ENTREGA FINAL TODO
import numpy as np
import copy


def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k),hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    ColMayorAFil = False
    if len(A) < len(A[0]):
        A = traspuesta(A)
        ColMayorAFil = True

    matriz = matmul(traspuesta(A), A)
    Avects, Avals = diagRH(matriz)

    if k == "max":
        k = len(Avals)

    hatSig = np.zeros(k)
    contador = 0
    for i in range(len(Avals)):
        if Avals[i][i] < tol:
            break
        if i < k:
            hatSig[i] = np.sqrt(Avals[i][i])
        contador += 1
    if contador < k:
        hatSig = hatSig[:contador]

    hatV = np.zeros((len(A[0]), k))
    for i in range(k):
        hatV[:, i] = Avects[:, i]

    hatU = np.zeros((len(A), k))
    for i in range(len(hatU[0])):
        hatU[:, i] = matmul(A, hatV)[:, i] / norma(matmul(A, hatV)[:, i], 2)

    if ColMayorAFil:
        return hatV, hatSig, hatU
    else:
        return hatU, hatSig, hatV

    # No esta testeada

    # else:
    #     matriz=matmul(A,traspuesta(A))
    #     Avects, Avals=diagRH(matriz)

    #     hatSig=np.zeros(k)
    #     contador=0
    #     for i in range(len(Avals)):
    #         if(Avals[i][i]>=tol and i<k):
    #             hatSig[contador]=np.sqrt(Avals[i][i])
    #             contador+=1
    #     if(contador<k):
    #         for i in range(k-len(hatSig)):
    #             hatSig[i]=0

    #     hatU=np.zeros((len(A[0]),k))
    #     for i in range(k):
    #         hatU[:,i] = Avects[i]
    #     hatU=traspuesta(hatU)

    #     hatV=matmul(traspuesta(hatU),A)/traspuesta(norma(traspuesta(hatU),A))


def svd_completa(A, tol=1e-15):
    V, Avals = diagRH(matmul(traspuesta(A), A), tol)  # matriz de autovect y de avals
    Sigma = []
    for i in range(len(Avals)):
        if Avals[i][i] >= tol:
            Sigma.append = np.sqrt(Avals[i][i])  # asigno a sigma sus valores singulares
    B = matmul(A, V)
    normalizado = []
    suma = np.zeros(len(B[0]))
    for i in traspuesta(B):  # por cada i-esima columna
        if norma(i, 2) != 0:
            normalizado = normalizado.append(i / norma(i, 2))  # normalizo
    while len(normalizado) < len(B):  # hasta que normalizado esté completo
        canonica = np.zeros(len(normalizado))
        for i in range(len(normalizado), len(B)):
            if norma(normalizado[len(normalizado) - i], 2) != 0:
                canonica[len(normalizado) - i] = 1  # creo el canonico
                suma = suma + proyectar(
                    canonica, normalizado[len(normalizado) - i]
                )  # voy sumando las proyecciones del canonico en los vectores del espacio ortonormal hasta ahora
                normalizado = normalizado.append(
                    normalizado[len(normalizado) - i] - suma
                )  # agrego un vector ortonormal
                canonica[len(normalizado) - i] = 0  # reinicio a ceros el canonico

    U = normalizado  # cambio de nombre para el return
    return U, Sigma, V

    # No esta testeada
