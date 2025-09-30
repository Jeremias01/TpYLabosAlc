import numpy as np
import sys
sys.path.append("./src")
from labo06_AVs import *


def run():
    print("Running tests for labo6...")
    ID2 = [[2, 0], [0, 0]]

    A1 = [[1/2, 1/2], [1/2, 1/2]]
    A2 = [[1, 1], [0, 1]]
    A4 = [[1, 0], [0, 2]]
    A5 = [[1, 0], [0, -2]]
    A3 = [[0, 1], [-1, 0]]
    
    A4AVMayor = metpot2k(A4, tol =10**(-15) ,K=1000)
    A5AVMayor = metpot2k(A5, tol =10**(-15) ,K=1000)
    A3AVMayor = metpot2k(A3, tol =10**(-15) ,K=1000)
    print(A5AVMayor, "\n\n")
    print(A3AVMayor)
    assert(np.allclose([2],[A4AVMayor[1]]))
    assert(np.allclose([-2],[A5AVMayor[1]]))
    print("\n\n")

    IDAVMayor = metpot2k(ID2, tol =10**(-15) ,K=1000)
    print(IDAVMayor)
    
    diagRH(A2)

    print("All tests for labo6 passed")

run()