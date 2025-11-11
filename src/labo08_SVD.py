import numpy as np
import copy

def svd_reducida(A, k = "max", tol = 1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k),hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    matriz=matmul(traspuesta(A), A)

    hatV, Avals=diagRH(matriz)

    hatSig=[]
    
    for i in range(len(Avals)):
        if(Avals[i][i]>=tol and i<=k):
            hatSig.append=sqr(Avals[i][i])
    if(len(hatSig)<k):
        for i in range(k-len(hatSig)):
            hatSig.append(0)

    

    hatU= normalize(matmul(A, hatV))

    return hatU, hatSig, hatV

    # No esta testeada


def svd_completa(A, tol = 1e-15):
    V,Avals=diagRH(A,tol) # matriz de autovect y de avals
    Sigma=[]
    for i in range(len(Avals)):
        if(Avals[i][i]>=tol):   
            Sigma.append=sqr(Avals[i][i])   # asigno a sigma sus valores singulares
    B=matmul(A,V)   
    normalizado=[]
    suma=np.zeros(len(B[0]))
    for i in traspuesta(B): # por cada i-esima columna
        if(norma(i,2)!=0):
            normalizado=normalizado.append(i/norma(i,2))    #normalizo
    while(len(normalizado)<len(B)):       # hasta que normalizado esté completo
        canonica=np.zeros(len(normalizado))
        for i in range(len(normalizado),len(B)):    
            if(norma(normalizado[i-len(normalizado)],2)!=0):
                canonica[i-len(normalizado)]=1      # creo el canonico
                suma=suma+proyectar(canonica,normalizado[i-len(normalizado)])       # voy sumando las proyecciones del canonico en los vectores del espacio ortonormal hasta ahora
                normalizado=normalizado.append(normalizado[i-len(normalizado)]-suma)        # agrego un vector ortonormal
                canonica[i-len(normalizado)]=0      # reinicio a ceros el canonico

    U=normalizado       #cambio de nombre para el return
    return U,Sigma,V

    # No esta testeada
