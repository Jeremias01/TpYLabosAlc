import numpy as np
import sys
sys.path.append("./src")
from labo06_AVs import *
from labo03_normas import normaExacta


def run():
    print("Running tests for labo6...")
    
    DP = [
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [1,2,3,4,5],
    ]
    DPP = [
        [7,0,0,0,0,0],
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
        [0,1,2,3,4,5],
    ]

    DPPP = expandirDiagonalPrincipalDesdeArriba(DP, 7)

    assert np.allclose(DPPP,DPP)

    
    
    
    
    
    ID2 = [[2, 0], [0, 0]]

    A1 = [[1/2, 1/2], [1/2, 1/2]]
    A2 = [[1, 1], [0, 1]]
    A4 = [[1, 0], [0, 2]]
    A5 = [[1, 0], [0, -2]]
    A3 = [[0, 1], [-1, 0]]
    
    A4AVMayor = metpot2k(A4, tol =10**(-15) ,K=1000)
    A5AVMayor = metpot2k(A5, tol =10**(-15) ,K=1000)
    A3AVMayor = metpot2k(A3, tol =10**(-15) ,K=1000)
    # print(A5AVMayor, "\n\n")
    # print(A3AVMayor)
    assert(np.allclose([2],[A4AVMayor[1]]))
    assert(np.allclose([-2],[A5AVMayor[1]]))
    # print("\n\n")

    # IDAVMayor = metpot2k(ID2, tol =10**(-15) ,K=1000)
    # print(IDAVMayor)
    

    diagRH(A2)





    # Test L06-metpot2k, Aval
    print("Empiezan Tests de catedra...")


    #### TESTEOS
    # Tests metpot2k

    S = np.vstack([
        np.array([2,1,0])/np.sqrt(5),
        np.array([-1,2,5])/np.sqrt(30),
        np.array([1,-2,1])/np.sqrt(6)
                ]).T

    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        D = np.diag(np.random.random(3)+1)*100
        A = S@D@S.T
        v,l,_ = metpot2k(A,1e-15,1e5)
        if np.abs(l - np.max(D))< 1e-8:
            exitos += 1
    assert exitos > 95


    #Test con HH
    exitos = 0
    for i in range(100):
        v = np.random.rand(9)
        #v = np.abs(v)
        #v = (-1) * v
        ixv = np.argsort(-np.abs(v))
        D = np.diag(v[ixv])
        I = np.eye(9)
        H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

        A = H@D@H.T
        v,l,_ = metpot2k(A, 1e-15, 1e5)
        #max_eigen = abs(D[0][0])
        if abs(l - D[0,0]) < 1e-8:         
            exitos +=1
    assert exitos > 95



    # Tests diagRH
    D = np.diag([1,0.5,0.25])
    S = np.vstack([
        np.array([1,-1,1])/np.sqrt(3),
        np.array([1,1,0])/np.sqrt(2),
        np.array([1,-1,-2])/np.sqrt(6)
                ]).T

    A = S@D@S.T
    SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
    print(D)
    print(DRH)
    assert np.allclose(D,DRH)
    assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



    # Pedimos que pase el 95% de los casos
    exitos = 0
    for i in range(100):
        A = np.random.random((4,4))
        A = 0.5*(A+A.T)
        S,D = diagRH(A,tol=1e-15,K=1e5)
        ARH = S@D@S.T
        e = normaExacta(ARH-A,p='inf')
        if e < 1e-5: 
            exitos += 1
    print(exitos)
    assert exitos >= 95

    print("All tests for labo6 passed")

