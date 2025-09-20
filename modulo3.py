import numpy as np

def maximo(l):
   res = l[0]
   for i in l:
    if i > res:
       res = i
   return res

def norma(x, p):
    if p != 'inf':
     suma = 0
     for i in range(len(x)):
        n = x[i]
        suma += abs(n)**p
     res = suma**(1/p)
     return res
    
    if p == 'inf':
       lista = []
       for i in range(len(x)):
          n = x[i].astype(float)
          lista.append(abs(x[i]))
       res = (maximo(lista))
    return (res)

def normaliza(X,p):
    res = []
    for i in range(len(X)):
        res.append(X[i]/norma(X[i],p))
    return res


def calcularAx(A,x):
  res = []
  for i in range(len(A)):
        v = 0
        for j in range(len(A[0])):
            v += x[j]*A[i][j]
        res.append(v)
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


def traspuesta(A):
    res = []
    for j in range(len(A[0])):       
        fila_trasp = []
        for fila in A:
            fila_trasp.append(fila[j])
        res.append(fila_trasp)
    return res


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
    norma_Inversa = normaMatMC(np.linalg.inv(A), p, p, Np)
    return norma_A*norma_Inversa


def condExacta(A,p):
  return normaExacta(A,p)*normaExacta(np.linalg.inv(A),p)
