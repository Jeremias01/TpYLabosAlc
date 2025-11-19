import numpy as np
from labo00_auxiliares import *


a=np.cos(0)
pi=np.pi

def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2 que rota un vector dado en un angulo theta
    """
    return np.array([
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)]])
    
    
    
def escala(vector):
    """
    Recibe una tira de números s y retorna una matriz cuadrada de n x n, donde n es el tamano de s. La matriz escala la componente i de un vector de Rn en un factor s[i]
    """
    res=np.zeros((len(vector),len(vector)))
    for i in range(len(vector)):
        res[i][i] = vector[i]
    return res



def rota_y_escala(theta,s):
    """
    Recibe un ángulo theta y una tira de números s, y retorna una matriz de 2 x 2 que rota el vector en un ángulo theta y luego lo escala en un factor s
    """
    res = matmul(escala(s), rota(theta))
    return res
    
 
def afin(theta,s,b):
    """
    Recibe un ángulo theta, una tira de números s (en R2), y un vector b en (R2) y retorna una matriz de 3 x 3 que rota el vector en un ángulo theta, luego lo escala en un factor s y por último lo muevo en un valor fijo b
    """
    res= rota_y_escala(theta, s)
    res = np.append(res,[[0,0]],0)
    res =  np.append(res,[[b[0]],[b[1]],[1]],1)
    return res


def trans_afin(v,theta,s,b):
    """
    Recibe un vector v (en R2), un ángulo theta, una tira de números s (en R2), y un vector b en (R2) y retorna el vector w resultante de aplicar la transformacion afin a v
    """
    mult = v + [1]
    return calcularAx(afin(theta,s,b), mult)[:-1]

    
    
    

    
