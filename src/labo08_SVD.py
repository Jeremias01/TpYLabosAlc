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
    # Problema: diahRH no es "reducida", tiene los AVals 0s y sus Avecs correspondientes. 
    # Se los vamos a ir quitando, contandolos y reduciendo siempre con dimension min(k,i)

    if k == "max":
        k = len(Avals)

    # nos quedamos con los sqrt(Avals) > 0. 
    hatSig = np.zeros(k)
    i = 0
    while i < k and Avals[i][i] > tol:
        hatSig[i] = np.sqrt(Avals[i][i])
        i += 1
    hatSig = hatSig[:i]

    # diagRH nos da autovectores de A^*A ortonormales para V! 
    hatV = Avects[:, :min(k,i)]


    hatU = np.zeros((len(A), min(k,i)))
    for i in range(len(hatU[0])):
        hatU[:, i] = matmul(A, hatV)[:, i] / hatSig[i]

    if ColMayorAFil:
        return hatV, hatSig, hatU
    else:
        return hatU, hatSig, hatV



def svd_completa(A, tol=1e-15):

    ColMayorAFil = False
    if len(A) < len(A[0]):
        A = traspuesta(A)
        ColMayorAFil = True

    V, Avals = diagRH(matmul(traspuesta(A), A), tol)  # matriz de autovect y de avals
    Sigma = np.zeros(len(Avals))
    for i in range(len(Avals)):
        if Avals[i][i] >= tol:               # si uno es cero, los que le siguen tambien
            Sigma[i] = np.sqrt(Avals[i][i])  # asigno a sigma sus valores singulares
    B = matmul(A, V)
    normalizado = []
    
    for i in traspuesta(B):  # por cada i-esima columna de B
        if norma(i, 2) != 0:
            normalizado = normalizado.append(i / norma(i, 2))  # la normalizo y la paso a filas de normalizado
    normalizado=traspuesta(normalizado)  # filas de noirmalizado a columnas 
    U = completar_ortonorm(normalizado,len(B))  # cambio de nombre para el return

    if ColMayorAFil:
        return V, Sigma, U          # si hay mas cols que filas, el proceso se invierte.
    else:
        return U, Sigma, V


def completar_ortonorm(matriz,num):
    mat=copy.deepcopy(matriz)
    suma = np.zeros(num)
    while len(mat) < num:  # hasta que mat esté completo
        canonica = np.zeros(num-len(mat))
        if(len(mat)==0):
            mat=traspuesta(identidad(num))
        else:
            for i in range(len(mat), num):            
                if (norma(mat[i-len(mat)], 2) != 0):
                    canonica[i-len(mat)] = 1  # creo el canonico
                    suma = suma + proyectar(
                        canonica, mat[i-len(mat)]
                    )  # voy sumando las proyecciones del canonico en los vectores del espacio ortonormal hasta ahora
                    mat = mat.append(
                        mat[i-len(mat)] - suma
                    )  # agrego un vector ortonormal

                    canonica[len(i-len(mat))] = 0  # reinicio a ceros el canonico

    return mat

# No esta testeada
