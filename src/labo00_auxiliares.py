import numpy as np
import sys
sys.path.append(".")
sys.path.append("../src")
sys.path.append("./src")

#TODO posiblemente convenga vectorizar esto. repetir el vector n veces, multiplciar con * con la matrix, sumar cada columna? no se si eso numpy lo vectorizaria bien. si si, aceleraría mucho matmul
def calcularAx(A,x):
    res = np.zeros(len(x))

    for i,row in enumerate(A):
        res[i] = prodint(row, x, conj=False)

    return res


def matmul(A,B):
    rowcount = len(A)
    colcount = len(B[0])

    assert len(B) == len(A[0])

    valorquecoincide = len(B)  
    res = np.zeros((rowcount, colcount))
    for r in range(rowcount):
        for c in range(colcount):
            res[r][c] = prodint( A[r],  B[:, c], conj=False)
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
    res = np.zeros((A.shape[1], A.shape[0]))
    for j, row in enumerate(A):       
        res[:,j] = row
    return res

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
    return np.array([row[::-1] for row in A][::-1])


def prodint(v1,v2, conj=True):  #prod int definido para vectores de la misma long
    if(len(v1)==len(v2)):
        if conj:
            v1 = np.conj(v1)
        v1conjv2 = v1*v2
        
        return np.sum(v1conjv2) #TODO preguntar si está bien usar sum, me imagino que si y que numpy lo implementa vectorizado


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
    return np.array([v])

def matCol(v):
    return traspuesta(np.array([v]))




def identidad(n):
    res = np.zeros((n,n))
    for i in range(n):
        res[i][i] = 1
    return res