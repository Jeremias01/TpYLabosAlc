import numpy as np

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
    return res
