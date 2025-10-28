import numpy as np
import sys
sys.path.append(".") 

from labo00_auxiliares import *


epsilon = 0.01 #(10**(-15))/2
def error(x, y):
  x = np.float64(x)
  y = np.float64(y)
    
  return abs(x-y)

def feq(x,y): # float equals
  return error(x,y) < epsilon # no se si absoluto o no

def error_relativo(x,y):
  x = np.float64(x)
  y = np.float64(y)
  n = abs(x-y)/abs(x)
  
  return n


def matricesIguales(A,B):
   if len(A) != len(B):
    return False
   
   for i in range(len(A)):
     for n in range(len(A)):
       if error(A[i] [n] , B[i] [n]) > epsilon:
         return False
       
   return True

# no esta en el labo pero seria raro ponerla en el labo00
def esSimetrica(A):
    return matricesIguales(A, traspuesta(A))
