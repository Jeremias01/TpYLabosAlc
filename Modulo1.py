import numpy as np
  
e = (10**(-15))/2
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
       if error(A[i] [n] , B[i] [n]) > e:
         return False
       
   return True

# Tests

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x, y), 0, atol=atol)

assert not sonIguales(1, 1.1)
assert sonIguales(1, 1 + np.finfo('float64').eps)
assert not sonIguales(1, 1 + np.finfo('float32').eps)
assert not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps)
assert sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3)

assert np.allclose(error_relativo(1, 1.1), 0.1)
assert np.allclose(error_relativo(2, 1), 0.5)
assert np.allclose(error_relativo(-1, -1), 0)
assert np.allclose(error_relativo(1, -1), 2)

assert matricesIguales(np.diag([1, 1]), np.eye(2))
assert matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2))
assert not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]]))
