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

normaMatMC([[1,2],[3,4],[5,6]], 2, 3, 10)
    
    
    

    
