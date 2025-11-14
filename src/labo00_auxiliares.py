import numpy as np
sys.path.append(".")
sys.path.append("../src")
sys.path.append("./src")
from labo01_errores_igualdad import *
from labo02_TLs_basicas import *
from labo03_normas import *

def calcularAx(A,x):
    res = []
    for row in A:
        res.append(0)
        for a1, x1 in zip(row,x):
            res[-1] += a1 * x1

    return np.array(res)


"""
def calcularAx(A,x):
  res = []
  for i in range(len(A)):
        v = 0
        for j in range(len(A[0])):
            v += x[j]*A[i][j]
        res.append(v)
  return res
"""

def matmul(A,B):
    rowcount = len(A)
    colcount = len(B[0])
    valorquecoincide = len(B) # == len(A[0]) 
    res = np.zeros((rowcount, colcount))
    for r in range(rowcount):
        for c in range(colcount):
            for k in range(valorquecoincide):
                res[r][c] += A[r][k] * B[k][c]
    return res

def triangularInferior(A):
    zeros=np.zeros((len(A),len(A[0])))
    for fila in range(len(A)):
        for col in range(len(A[0])):
            if(fila>col):
                zeros[fila][col]=A[fila][col]
    return zeros

def triangularSuperior(A):
    zeros=np.zeros((len(A),len(A[0])))
    for fila in range(A):
        for col in range(len(A)):
            if(fila<col):
                zeros[fila][col]=A[fila[col]]
    return zeros

def maximo(l):
   res = l[0]
   for i in l:
    if i > res:
       res = i
   return res

def traspuesta(A):
    res = []
    for j in range(len(A[0])):       
        fila_trasp = []
        for fila in A:
            fila_trasp.append(fila[j])
        res.append(fila_trasp)
    return np.array(res)

def traspuestaPorOtraDiagonal(A):
    mid=np.zeros((len(A),len(A[0])))
    res=np.zeros((len(A),len(A[0])))
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            mid[i][j]=A[j][len(A[0])-1-i]

    mid=traspuesta(mid)

    for i in range(len(A)):
        for j in range(len(A[0])):
            res[i][j]=mid[len(A)-1-j][i]

    return res

def rotar180(A):
    return [row[::-1] for row in A][::-1]


def prodint(v1,v2):  #prod int definido para vectores de la misma long
    if(len(v1)==len(v2)):
        res=0
        for i in range(len(v1)):
            res+=np.conj(v1[i])*v2[i]
        return res


def cuadrada(A):
    return len(A)>0 and len(A) == len(A[0])  


def expandirDiagonalPrincipalDesdeArriba(D, zerozero):
    D = np.insert(D, 0, np.zeros((1,len(D))) ,0)
    D = np.insert(D, 0, np.zeros((1,len(D))),1)
    D[0][0] = zerozero
    return D

def sign(n):
    if n == 0:
        return 0
    if n > 0:
        return 1
    if n < 0:
        return -1 

def matFila(v):
    return [v]

def matCol(v):
    return traspuesta([v])


def proyectar(v,u):
    if(norma(v,2)!=0 and norma(u,2)!=0):
        return prodint(prodint(v,u)/prodint(u,u),u)
    else: return np.zeros(len(v))

def identidad(n):
    res = np.zeros((n,n))
    for i in range(n):
        res[i][i] = 1
    return res