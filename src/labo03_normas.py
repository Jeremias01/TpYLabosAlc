
"""
def norma(x,p):
    if p == 'inf':
        maxv = abs(x[0])
        for item in x:
            if abs(item) > maxv:
                maxv = abs(item)
        return maxv
    suma=0
    for i in x:
        suma+=abs(i)**p
    return (suma)**(1/p)


norma([-1,1],'inf')


def normaliza(X,p):
    res = []
    for item in X:
        res.append(norma(item,p))
    return np.array(res)

def normaMatMC(A,q,p,Np):
    randxs = np.random.rand(Np,len(A[0]))
    maxv = norma(calcularAx(A, randxs[0]),p)/norma(randxs[0],q)
    for x in randxs:
        n = norma(calcularAx(A, x),p)/norma(x,q)
        if n>maxv: maxv=n
    return maxv
"""
    
    
    

    



import numpy as np
import sys
sys.path.append(".") 

from labo00_auxiliares import *
from labo04_LU import inversa

# calcula norma p de un vector x, vale usar ´inf´
def norma(x, p):
    if p != 'inf':
        return np.sum(np.power(np.abs(x), p))**(1/p)
    
    if p == 'inf':
        return np.max(np.abs(x))
    

# normaliza una lista de vectores
def normaliza(X,p):
    res = []
    for i in range(len(X)):
        res.append(X[i]/norma(X[i],p))
    return res





def normaMatMC(A, q, p, Np):
    norma_max = 0
    vector_max = None


    for _ in range(Np):
        x_random = np.random.randn(len(A[0]))
        x_normalizado = x_random / norma(x_random, p)

        nuevo_vector = calcularAx(A,x_normalizado)
        norma_final = norma(nuevo_vector,q)

        if norma_final > norma_max:
            norma_max = norma_final
            vector_max = x_normalizado

    return norma_max, vector_max




def suma_filas(A):
   res = []
   for i in range(len(A)):
        c = 0
        for elem in A[i]:
            c += abs(elem)
        res.append(c)
   return res

def suma_columnas(A):
   return suma_filas(traspuesta(A))


def normaExacta(A, p=[1, 'inf']):
   if p == 1:
    return maximo(suma_columnas(A))
    

   if p == 'inf':
    return maximo(suma_filas(A))

  
    

def condMC(A,p, Np):
    norma_A = normaMatMC(A, p, p, Np)
    norma_Inversa = normaMatMC(inversa(A), p, p, Np)
    return norma_A[0] * norma_Inversa[0]


def condExacta(A,p):
  return normaExacta(A,p)*normaExacta(inversa(A),p)


def proyectar(v,u):
    if(norma(v,2)!=0 and norma(u,2)!=0):
        return prodint(prodint(v,u)/prodint(u,u),u)
    else: return np.zeros(len(v))