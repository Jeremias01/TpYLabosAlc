#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
import sys
sys.path.append(".") # poner path
import labo0

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for iter in range(n):
        pivot = Ac[iter][iter]
        if pivot == 0:
            return None # TODO que hay que hacer
        for fila in range(iter+1,n):
            L_i_inv = Ac[fila][iter] / pivot; cant_op += 1
            Ac[fila][iter] = L_i_inv 
            Ac[fila][iter+1:n] = -L_i_inv * Ac[iter][iter+1:n] + Ac[fila][iter+1:n] ; cant_op += (n-(iter+1))*3
    
    L = labo0.triangularInferior(Ac) + np.identity(n) ; cant_op += n**2

    U = Ac - labo0.triangularInferior(Ac) 



                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

#if __name__ == "__main__":
#    main()


A = np.array([
    [2,1,2,3],
    [4,3,3,4],
    [-2,2,-4,-12],
    [4,1,8,-3],
]   )


elim_gaussiana(A)

# NO FUNCIONA
def calcularUxEQy(U, b):
    n = len(b)
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        suma_valores_anteriores = 0
        for j in range(i,n-1):
            suma_valores_anteriores += x[j] * U[i][j]
                
        x[i] = (b[i] - suma_valores_anteriores)/U[i][i]
    return x


# def calcularLyEQb(L, b):

U = [
    [1, -3, 1],
    [0, -4, 1],
    [0,0,-1]
]

b = [2,0,1]

print(calcularUxEQy(U,b))