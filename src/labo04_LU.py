#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np
import sys
sys.path.append(".") # poner path
import labo00_auxiliares

def calculaLU(A):
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
    
    L = labo00_auxiliares.triangularInferior(Ac) + np.identity(n) ; cant_op += n**2

    U = Ac - labo00_auxiliares.triangularInferior(Ac) 



                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op


A = np.array([
    [2,1,2,3],
    [4,3,3,4],
    [-2,2,-4,-12],
    [4,1,8,-3],
]   )



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

