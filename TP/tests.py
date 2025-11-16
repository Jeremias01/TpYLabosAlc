import numpy as np
import sys
from alc import *



A=[[4,12,-16],[12,37,-43],[-16,-42-98]]
Lreal=[[2,0,0],[6,1,0],[-8,5,3]]





assert np.allclose(Lreal, cholesky(A), 'cholesky no da lo correcto')