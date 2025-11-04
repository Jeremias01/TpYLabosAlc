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
