import numpy as np
import sys
sys.path.append(".") 

from labo00_auxiliares import *


epsilon = (10**(-14))
def error(x, y):
    x = np.float64(x)
    y = np.float64(y)
    
    return abs(x-y)

def feq(x,y): # float equals
    return error(x,y) < epsilon # no se si absoluto o no

def error_relativo(x,y):
    """
    Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64
    """
    x = np.float64(x)
    y = np.float64(y)
    n = abs(x-y)/abs(x)
  
    return n


def matricesIguales(A,B, atol=epsilon):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    if A.shape != B.shape:
        return False
    fils,cols = A.shape
    
    for row in range(fils):
        for col in range(cols):
            if error(A[row][col] , B[row][col]) > atol:
                return False
       
    return True

# no esta en el labo pero seria raro ponerla en el labo00
def esSimetrica(A, atol=epsilon):
    return matricesIguales(A, traspuesta(A), atol)
