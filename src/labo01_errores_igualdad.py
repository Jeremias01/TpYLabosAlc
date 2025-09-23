import numpy as np
  
epsilon = (10**(-15))/2
def error(x, y):
  x = np.float64(x)
  y = np.float64(y)
    
  return abs(x-y)
  
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

