import numpy as np
import sys
import os
import copy
from datetime import datetime

###############################################################################
###############################################################################
###############################################################################
################################# AUXILIARES ##################################
###############################################################################
###############################################################################
###############################################################################

def calcularAx(A,x):
    res = np.zeros(A.shape[0])

    for i,row in enumerate(A):
        res[i] = prodint(row, x, conj=False)

    return res

def calcular_xtA(A,x):
    res = np.zeros(A.shape[1])

    for i in range(A.shape[1]):
        res[i] = prodint(A[:,i], x, conj=False)

    return res


def matmul(A,B):
    rowcount = len(A)
    colcount = len(B[0])

    assert len(B) == len(A[0])
    
    res = np.zeros((rowcount, colcount))
    for r in range(rowcount):
        for c in range(colcount):
            res[r][c] = prodint( A[r],  B[:, c], conj=False)
    return res

def triangularInferior(A):
    zeros=np.zeros((len(A),len(A[0])))
    for fila in range(len(A)):
        for col in range(len(A[0])):
            if(fila>col):
                zeros[fila][col]=A[fila][col]
    return zeros

def triangularSuperior(A):
    zeros=np.zeros((len(A),len(A[0])))
    for fila in range(A):
        for col in range(len(A)):
            if(fila<col):
                zeros[fila][col]=A[fila[col]]
    return zeros

def maximo(l):
   res = l[0]
   for i in l:
    if i > res:
       res = i
   return res

def traspuesta(A):
    res = np.zeros((A.shape[1], A.shape[0]))
    for j, row in enumerate(A):       
        res[:,j] = row
    return res

def traspuestaPorOtraDiagonal(A):
    mid=np.zeros((len(A),len(A[0])))
    res=np.zeros((len(A),len(A[0])))
    
    for i in range(len(A)):
        for j in range(len(A[0])):
            mid[i][j]=A[j][len(A[0])-1-i]

    mid=traspuesta(mid)

    for i in range(len(A)):
        for j in range(len(A[0])):
            res[i][j]=mid[len(A)-1-j][i]

    return res

def rotar180(A):
    return np.array([row[::-1] for row in A][::-1])


def prodint(v1,v2, conj=True):  #prod int definido para vectores de la misma long
    if(len(v1)==len(v2)):
        if conj:
            v1 = np.conj(v1)
        v1conjv2 = v1*v2
        
        return np.sum(v1conjv2) 


def cuadrada(A):
    return len(A)>0 and len(A) == len(A[0])  


def expandirDiagonalPrincipalDesdeArriba(D, zerozero):
    D = np.insert(D, 0, np.zeros((1,len(D))) ,0)
    D = np.insert(D, 0, np.zeros((1,len(D))),1)
    D[0][0] = zerozero
    return D

def sign(n):
    if n == 0:
        return 0
    if n > 0:
        return 1
    if n < 0:
        return -1 

def matFila(v):
    return np.array([v])

def matCol(v):
    return traspuesta(np.array([v]))




def identidad(n):
    res = np.zeros((n,n))
    for i in range(n):
        res[i][i] = 1
    return res


###############################################################################
###############################################################################
###############################################################################
############################ IGUALDAD ##### LABO 1 ############################
###############################################################################
###############################################################################
###############################################################################

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


###############################################################################
###############################################################################
###############################################################################
############################ TLS ########## LABO 2 ############################
###############################################################################
###############################################################################
###############################################################################

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



###############################################################################
###############################################################################
###############################################################################
############################ NORMAS ####### LABO 3 ############################
###############################################################################
###############################################################################
###############################################################################

# calcula norma p de un vector x, vale usar ´inf´
def norma(x, p):
    if p != 'inf':
        # No tenemos permitdo usar np.power para vectorizar las potencias. entonces np.sum no sirve para vectorizar.
       #  return np.sum(np.power(np.abs(x), p))**(1/p)
    
        sum = 0
        for item in x:
            sum += np.abs(item) ** p
        return sum ** (1/p)        
        
    
    if p == 'inf':
        return np.max(np.abs(x))
    

# normaliza una lista de vectores
def normaliza(X,p):
    res = []
    for i in range(len(X)):
        res.append(X[i]/norma(X[i],p))
    return res





def normaMatMC(A, q, p, Np):
    norma_max = 0
    vector_max = None


    for _ in range(Np):
        x_random = np.random.randn(len(A[0]))
        x_normalizado = x_random / norma(x_random, p)

        nuevo_vector = calcularAx(A,x_normalizado)
        norma_final = norma(nuevo_vector,q)

        if norma_final > norma_max:
            norma_max = norma_final
            vector_max = x_normalizado

    return norma_max, vector_max




def suma_filas(A):
   res = []
   for i in range(len(A)):
        c = 0
        for elem in A[i]:
            c += abs(elem)
        res.append(c)
   return res

def suma_columnas(A):
   return suma_filas(traspuesta(A))


def normaExacta(A, p=[1, 'inf']):
   if p == 1:
    return maximo(suma_columnas(A))
    

   if p == 'inf':
    return maximo(suma_filas(A))

  
    

def condMC(A,p, Np):
    norma_A = normaMatMC(A, p, p, Np)
    norma_Inversa = normaMatMC(inversa(A), p, p, Np)
    return norma_A[0] * norma_Inversa[0]


def condExacta(A,p):
  return normaExacta(A,p)*normaExacta(inversa(A),p)


def proyectar(v,u):
    if(norma(v,2)!=0 and norma(u,2)!=0):
        return prodint(prodint(v,u)/prodint(u,u),u)
    else: return np.zeros(len(v))

###############################################################################
###############################################################################
###############################################################################
############################ LU ########### LABO 4 ############################
###############################################################################
###############################################################################
###############################################################################


def calculaLU(A):
    """
    Calcula la factorizacion LU de la matriz A y retorna las matrices L
    y U, junto con el numero de operaciones realizadas. En caso de
    que la matriz no pueda factorizarse retorna Nones.
    """
    cant_op = 0
    A = np.array(A, dtype=float)
    m = A.shape[0]
    n = A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR

    for iter in range(n):
        pivot = Ac[iter][iter]
        if pivot == 0:
            return None, None, 0
        for fila in range(iter+1,n):
            L_i_inv = Ac[fila][iter] / pivot                                            ; cant_op += 1
            Ac[fila][iter] = L_i_inv 
            Ac[fila][iter+1:n] = -L_i_inv * Ac[iter][iter+1:n] + Ac[fila][iter+1:n]     ; cant_op += (n-(iter+1))*2 # para mi era *3 no *2 pero fallan tests 
    
    L = triangularInferior(Ac) + identidad(n)                       ; # estas no cuenan cant_op += n**2, si no fallan tests

    U = Ac - triangularInferior(Ac) 

    
    return L, U, cant_op




def res_tri(L, b, inferior = True) :
    """
    Resuelve el sistema Lx = b, donde L es triangular. Se puede indicar
    si es triangular inferior o superior usando el argumento
    inferior (por default asumir que es triangular inferior).
    """
    if not inferior:
        L = rotar180(L)
        b = b[::-1]
    n = len(b)
    x = np.zeros(n)
    for i, row in enumerate(L):
        x[i] = (b[i] - prodint(row[:i], x[:i]) )/row[i]
    
    
    return x if inferior else x[::-1]

def res_tri_mat(L, B, inferior = True):
    """
    Resuelve el sistema LX = B, 
    Recibe L triangular, superior o inferior segun parametro inferior, y B.
    Devuelve X, la matriz solucion. 
    """
    print(type(L), type(B))
    if L.shape[0] != B.shape[0]:
        return None 
    res = np.zeros((L.shape[0], B.shape[1]))
    for i,col in enumerate(traspuesta(B)):
        res[:,i] = res_tri(L, col, inferior)
    return res


def res_LU(L,U, b):
    y = res_tri(L,b)
    x = res_tri(U,y, inferior=False)
    return x

def res_LU_mat(L,U, B):
    Y = res_tri_mat(L,B)
    X = res_tri_mat(U,Y, inferior=False)
    return X


def inversa(A):
    """
    Calcula la inversa de A empleando la factorizacion LU
    y las funciones que resuelven sistemas triangulares
    retorna None si no es inversible
    """
    A = np.array(A, dtype=float)
    L,U,_ = calculaLU(A)
    if L is None or U is None:
        return None
    res = identidad(A.shape[0])
    for i in range(len(res)):
        res[i] = res_LU(L,U,res[i])
    return traspuesta(res) 

def calculaLDV(A):
    """
    Calcula la factorizacion LDV de la matriz A, de forma tal que A =
    LDV, con L triangular inferior, D diagonal y V triangular
    superior. En caso de que la matriz no pueda factorizarse
    retorna None.
    Ademas devuelve la cantidad de operaciones realizadas
    Devuleve 
    """
    L,U,opA = calculaLU(A)
    if L is None or U is None:
        return None,None,None,0
    V,D,opU = calculaLU(traspuesta(U))
    return L,D,traspuesta(V),opA+opU

def esSDP(A, atol=1e-8) :
    """
    Checkea si la matriz A es simetrica definida positiva (SDP) usando
    la factorizacion LDV.
    """
    if not esSimetrica(A, atol):
        return False
    L,D,V,_ = calculaLDV(A)
    if L is None or D is None or V is None:
        return False
    for i in range(A.shape[0]):
        if A[i][i] <= 0:
            return False

    return True


###############################################################################
###############################################################################
###############################################################################
############################ QR ########### LABO 5 ############################
###############################################################################
###############################################################################
###############################################################################


"""
A una matriz de n x n 
tol la tolerancia con la que se filtran elementos nulos en R
retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
Si la matriz A no es de n x n, debe retornar None
"""

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    #if not A.shape[1] >= A.shape[0]: # el apunte del TP pide mas filas que columnas. la toerica decia mas columnas que filas. esta es la que funciona
    #    return None
    m = len(A)
    n = len(A[0])
    

    contador=0
    QT=np.zeros((n,m)) 
    R=np.zeros((n,n)) # R=np.zeros((k,n)) si no queres los 0s de mas
    AColumnas = traspuesta(A).astype(np.float64)

    R[0][0]=norma(AColumnas[0],2)

    # definimos primer columna de Q como la primera de A normalizada
    if feq(R[0][0] , 0):
        QT[0] = np.zeros((n))
    else:
        QT[0]=AColumnas[0]/R[0][0]

    # por cada columna que queda, 
    for j in range(1,n):
        if j % 100 == 0:
            print(f"ortonormalizando {j}-esimo vector de {n} a las {datetime.now().time()}")
        Qj=AColumnas[j]
        for k in range(0,j):
            R[k][j]=prodint(QT[k],Qj)
            Qj+=-R[k][j]*QT[k]
        R[j][j]=norma(Qj,2)

        if feq(R[j][j] , 0):
            QT[j] = np.zeros((n,))
        else:
            QT[j]=Qj*(1/R[j][j])
    
    
    if retorna_nops :
        return traspuesta(QT),R,contador
    else: 
        return traspuesta(QT),R 



def houseHolder(u):
    # precondición: u tal que ||u||_2 = 1
    mMenosK = len(u) 
    return identidad(mMenosK) - 2 * uvt(u,u) 

# u @ v^t
def uvt(u,v):
    return traspuesta(u * matCol(v).repeat(len(u), axis=1))

# A @ houseHolder(u)  pero mas eficiente
def matmulHouseHolderDerecha(A,u):
    return A - uvt(2 * calcularAx(A, u), u)

def matmulHouseHolderDerechaMenor(A,u):
    Amenor = A[:, A.shape[0]-len(u):]
    res_menor = matmulHouseHolderDerecha(Amenor,u)
    A[:, A.shape[0]-len(u):] = res_menor
    return A


#  houseHolder(u) @ A  pero mas eficiente
def matmulHouseHolderIzquierda(u,A):
    return A - uvt(2 * u, calcular_xtA(A, u))
    
def matmulHouseHolderIzquierdaMenor(u,A):
    Amenor = A[A.shape[0]-len(u):, :]
    res_menor = matmulHouseHolderIzquierda(u,Amenor)
    A[A.shape[0]-len(u):, :] = res_menor
    return A

def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    if ( m := len(A) ) == 0 or m < ( n := len(A[0])):
        return None
    
    R = A.astype(np.float64)
    Q = identidad(m)
    for k in range(0,n):
        if k % 100 == 0:
            print(f"householderizando {k}-esima sumbatriz de {n} a las {datetime.now().time()}")

        x = R[k:, k]
        alpha = - sign(x[0]) * norma(x, 2)     # doble negacion redundante??
        u = x - alpha * identidad(m-k)[0]
        if (unorma := norma(u, 2)) > tol:
            u = u / unorma
            #Hk =  houseHolder(u)
            #for iter in range(k): 
            #    Hk = expandirDiagonalPrincipalDesdeArriba(Hk, 1)
            #
            #R = matmul(Hk,R)
            #Q = matmul(Q,traspuesta(Hk))       
            R = matmulHouseHolderIzquierdaMenor(u,R)
            Q = matmulHouseHolderDerechaMenor(Q,u)


    #borramos filas y columnas redundantes antes de devolver
    return Q[:,:n],R[:n,:]



def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == "RH":
        return QR_con_HH(A,tol=1e-12)
    elif metodo == 'GS':
        return QR_con_GS(A,tol=1e-12)
        

###############################################################################
###############################################################################
###############################################################################
############################ AVs ########## LABO 6 ############################
###############################################################################
###############################################################################
###############################################################################


def aplicarPotenciaUnaVez(A,v):
    vSombrero = calcularAx(A, v)
    norm = norma(vSombrero,2)
    if norm == 0:
        return 0
    vSombrero = vSombrero / norm
    return vSombrero

def aplicarPotenciaDosVeces(A,v):
    
    vSombrero = aplicarPotenciaUnaVez(A,aplicarPotenciaUnaVez(A,v))

    e = prodint(vSombrero, v)
    return vSombrero, e

def metpot2k(A, tol =10**(-15) ,K=1000):
    """
    A: una matriz de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    Retorna vector v, autovalor lambda y numero de iteracion realizadas k.
    """
    n = len(A)
    if not cuadrada(A):
        return None
    v = np.random.randn(n)
    vSombrero, e = aplicarPotenciaDosVeces(A, v)
    for k in range(int(K)):
        if abs(abs(e)-1)<=tol:
            break
        v = vSombrero
        vSombrero, e = aplicarPotenciaDosVeces(A, v)

    aVal = prodint(vSombrero, calcularAx(A, vSombrero))
    err = e-1

    return vSombrero, aVal, k





def diagRH(A, tol =1e-8 ,K=1000):
    """
    A: una matriz simetrica de n x n.
    tol: la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K: el numero maximo de iteraciones a realizarse.
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S. T
    Si la matriz A no es simetrica, debe retornar None.
    """

    ## DESHABILITO ESTOS CHEQUEOS PORQUE OCUPAN MUCHO TIEMPO
    if not cuadrada(A):
        return None
    #if not (esS:=esSimetrica(A, tol)):
    #    # si lo dejo haciendo cuentas grandes no quiero perderlo todo pq estaba mal la toleranica
    #    print("WARNING: no se cumple la toleranicia pedida" )
    #    # return None
    n = len(A)
    if n % 20 == 0:
        print(f"diagonalizando {n}-esima sumbatriz a las {datetime.now().time()}")


    v,aVal,_ = metpot2k(A,tol,K)
    e1_menos_v = identidad(n)[0] - v
    # Hv = houseHolder( e1_menos_v / norma(e1_menos_v, 2) )
    v_para_householder = e1_menos_v / norma(e1_menos_v, 2)
    if n == 2:
        S_matriz_avecs = houseHolder(v_para_householder)
        D_matriz_avals = matmulHouseHolderIzquierda(v_para_householder, matmulHouseHolderDerecha(A, v_para_householder))
    else:
        B = matmulHouseHolderIzquierda(v_para_householder, matmulHouseHolderDerecha(A, v_para_householder))
        ASombrero = B[1:,1:]
        SSombrero,DSombero  = diagRH(ASombrero, tol, K)
        D_matriz_avals = expandirDiagonalPrincipalDesdeArriba(DSombero, aVal)  
        S_matriz_avecs = matmulHouseHolderIzquierda(v_para_householder,
            expandirDiagonalPrincipalDesdeArriba(SSombrero, 1))      

    if n % 20 == 0:
        print(f"listo diagonalizando {n}-esima sumbatriz a las {datetime.now().time()}")


    return S_matriz_avecs, D_matriz_avals


###############################################################################
###############################################################################
###############################################################################
############################ Markov ####### LABO 7 ############################
###############################################################################
###############################################################################
###############################################################################


def transiciones_al_azar_continuas(n):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el
    intervalo [0, 1]
    """
    m = np.abs(np.random.rand(n,n))
    for i in range(n):
        m[:, i] = m[:, i] / norma(m[:, i],1)
    return m




def transiciones_al_azar_uniforme(n, thres):
    """
    n: la cantidad de filas (columnas) de la matriz de transición.
    thres: probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas. El elemento i, j es distinto de
    cero si el número generado al azar para i, j es menor o igual a thres. Todos los
    elementos de la columna $j$ son iguales (a 1 sobre el número de elementos distintos
    de cero en la columna).
    """
    m = np.zeros((n,n)) #np.abs(np.random.rand(n,n))
    for i in range(n):
        while norma(m[:, i],1) == 0:
            m[:, i] = (np.abs(np.random.rand(n))>thres).astype(np.float64) # dudoso
        m[:, i] = m[:, i] / norma(m[:, i],1)
    return m



def nucleo(A, tol=1e-15):
    """
    A: una matriz de m x n
    tol: la tolerancia para asumir que un vector está en el núcleo.
    Calcula el núcleo de la matriz A diagonalizando la matriz traspuesta (A) * A (* la
    multiplicación matricial), usando el medodo diagRH. El núcleo corresponde a los
    autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestión, como una matriz de n x k, con k el numero de
    autovectores en el núcleo.
    """

    
    AtA = matmul( traspuesta(A) , A )
    S_matriz_avecs, D_matriz_avals = diagRH(AtA)
    res = []
    for i in range(len(S_matriz_avecs)):
        if D_matriz_avals[i][i] <= tol:
            res.append(S_matriz_avecs[:,i])
    print(res)
    return traspuesta(np.array(res)) if res != [] else np.array([])



def crear_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con índices i, lista con índices
    j, y lista con valores A_ij de la matriz A. Tambien las dimensiones de la matriz a
    traves de m_filas y n_columnas. Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores
    distintos de cero. Retorna una lista con:
    - Diccionario { (i, j): A_ij } que representa los elementos no nulos de la matriz A. Los
    elementos con modulo menor a tol deben descartarse por default.
    - Tupla (m_filas, n_columnas) que permita conocer las dimensiones de la matriz.
    """
    res = dict()
    if listado != []:
        filas, columnas, valores = listado
        for fila, columna, valor in zip(filas, columnas, valores):
            if valor > tol:
                res[(fila, columna)] = valor
    return [res, (m_filas, n_columnas)]


def multiplicar_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crear_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """
    Ashape = A[1]
    Amat = A[0]
    res = np.zeros(Ashape[0])
    for i in range(Ashape[0]):
        for j in range(Ashape[1]):
            if (i,j) in A[0]:
                res[i] += A[0][(i,j)] * v[j]
    return res


###############################################################################
###############################################################################
###############################################################################
############################ SVD ########## LABO 8 ############################
###############################################################################
###############################################################################
###############################################################################


def svd_reducida(A, k="max", tol=1e-8):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna hatU (matriz de m x k),hatSig (vector de k valores singulares) y hatV (matriz de n x k)
    """

    ColMayorAFil = False
    if len(A) < len(A[0]):
        A = traspuesta(A)
        ColMayorAFil = True

    matriz = matmul(traspuesta(A), A)
    Avects, Avals = diagRH(matriz, tol)
    # Problema: diahRH no es "reducida", tiene los AVals 0s y sus Avecs correspondientes. 
    # Se los vamos a ir quitando, contandolos y reduciendo siempre con dimension min(k,i)

    if k == "max":
        k = len(Avals)

    # nos quedamos con los sqrt(Avals) > 0. 
    hatSig = np.zeros(k)
    i = 0
    while i < k and Avals[i][i] > tol:
        hatSig[i] = np.sqrt(Avals[i][i])
        i += 1
    hatSig = hatSig[:i]

    # diagRH nos da autovectores de A^*A ortonormales para V! 
    hatV = Avects[:, :min(k,i)]


    hatU = np.zeros((len(A), min(k,i)))
    AhatV = matmul(A, hatV)
    for i in range(len(hatU[0])):
        hatU[:, i] = AhatV[:, i] / hatSig[i]

    if ColMayorAFil:
        return hatV, hatSig, hatU
    else:
        return hatU, hatSig, hatV

###############################################################################
###############################################################################
###############################################################################
############################ FUNCIONES DEL TP #################################
###############################################################################
###############################################################################
###############################################################################

def cargarDataset(carpeta):
    """
    carpeta: un string con dirección a la carpeta con datos de entrenamiento
    retorna matrices Xt, Yt, Xv, Yv. 
    Las matrices (Xt, Yt) corresponden a los ejemplos de entrenamiento conteniendo los embeddings de
    los gatos y perros juntos. Cada columna de la matriz corresponde a un embedding. La matriz Yt
    debe generarse a partir de la lectura de los archivos de entrenamiento. Cada columna de Yt tiene
    2 elementos valiendo 0 o 1 dependiendo de la clase a la que pertenece el embedding. 
    Por ejemplo un yi de un gato debería ser yi = [1, 0]T , y otro de un perro: yj = [0, 1]T .
    """
    
    XvCats = np.load(f'{os.getcwd()}/{carpeta}/val/cats/efficientnet_b3_embeddings.npy')
    XtCats = np.load(f'{os.getcwd()}/{carpeta}/train/cats/efficientnet_b3_embeddings.npy')
    XvDogs = np.load(f'{os.getcwd()}/{carpeta}/val/dogs/efficientnet_b3_embeddings.npy')
    XtDogs = np.load(f'{os.getcwd()}/{carpeta}/train/dogs/efficientnet_b3_embeddings.npy')
    Xv = np.concatenate((XvDogs, XvCats), 1)    
    Xt = np.concatenate((XtDogs, XtCats), 1)    

    YvCats = np.concatenate(tuple([[1],[0]] for x in range(XvCats.shape[1])),1)
    YvDogs = np.concatenate(tuple([[0],[1]] for x in range(XvDogs.shape[1])),1)
    YtCats = np.concatenate(tuple([[1],[0]] for x in range(XtCats.shape[1])),1)
    YtDogs = np.concatenate(tuple([[0],[1]] for x in range(XtDogs.shape[1])),1)

    Yv = np.concatenate((YvDogs, YvCats), 1)    
    Yt = np.concatenate((YtDogs, YtCats), 1)    


    return Xt, Yt, Xv, Yv


def cargarDatasetReducido(carpeta, filas, columnas):
    """
    columnas debe ser mayor a 2*filas para que la matriz que devuelva sea compatible con los otros metodos
    para probar con un subconjunto del dataset. filas es cuantas filas de cada embedding, y columnas cuantos embeddings de perros y de gatos
    devuelve matriz de filas x 2*columnas.
    """
    Xt, Yt, Xv, Yv = cargarDataset(carpeta)
    return (np.concatenate((Xt[:filas,:columnas], Xt[:filas,-columnas:]),1),
            np.concatenate((Yt[:,:columnas] , Yt[:,-columnas:]),1),
            np.concatenate((Xv[:filas,:columnas] , Xv[:filas,-columnas:]),1),
            np.concatenate((Yv[:,:columnas] , Yv[:,-columnas:]),1),
        )

# Esta funcion obtiene la matriz L (Lower) utilizada al factorizar por Cholesky de la forma A = L*L^t
# Acepta solo matrices A diagonales, que es nuestro caso de uso
def cholesky(A):
    L = np.zeros((len(A),len(A[0])))
 
    for columna in range((len(A))):
        if columna % 100 == 0:
            print(f"choleskizando {columna}-esima columna a las {datetime.now().time()}")
        
        #suma = np.sum(np.power(L[columna],2))
        suma = 0
        for i in range(columna):
            suma += (L[columna][i])**2
        
        diagonal = np.sqrt(A[columna][columna]- suma)
        L[columna][columna] = diagonal

        for fila in range(columna+1, len(A)):
            suma = np.sum(L[fila][:columna] * L[columna][:columna])
            
            L[fila][columna] = (A[fila][columna] - suma) / L[columna][columna]

    return L

def pinvEcuacionesNormales(X,L, Y):
    """
    Resuelve el cálculo de los pesos utilizando las ecuaciones normales para
    la resolución de la psetargetsudo-inversa usando el algoritmo 1 y descomposición cholesky. 
    X: la matriz original
    L: la matriz de Cholesky de XX^t 
    Y: la matriz de  de entrenamiento.
    retorna cálculo de los pesos W
    """
    # el enunciado pide usar  V X X^t    = X^t.
    #                        (V X X^t)^t = X
    #                         X X^t V^t = X
    #                         L L^t V^t = X
    
    Vt = res_LU_mat(L, traspuesta(L), X)
    V = traspuesta(Vt)
    W = matmul(Y,V)
    return W


def pinvSVD(U, S, V, Y, tol=1e-8):
    """ 
    Obtiene los pesos utilizando la Descomposición
    en Valores Singulares para la resolución de la pseudo-inversa usando el algoritmo 2. 
    U: Matriz de autovectores por izquierda de SVD reducida
    S: Matriz Sigma de valores singulares 
    V: Matriz de autovectores por derecha de SVD reducida
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    # Queremos un vector S_inv con valores singulares > 0

    k = min(S.shape[0],S.shape[1])
    S_inv = np.zeros(k)
    i = 0
    while i < k and S[i][i] > tol:
        S_inv[i] = S[i][i] ** -1
        i += 1
    S_inv = S_inv[:i]


    # Como tenemos S_inv vector que representa una matriz diagonal, podemos hacer mas rápida la multiplicación
    VS_inv = np.zeros(V.shape)
    for i in range(V.shape[1]):
        VS_inv[:, i] = V[:, i] * S_inv[i]
    
    A_pseudoinv = matmul(VS_inv, traspuesta(U))
    #assert esPseudoInversa(U @ np.diag(S) @ traspuesta(V), A_pseudoinv, 1e-8)
    
    W = matmul(Y,A_pseudoinv)

    return W
    

def pinvQR(Q,R,Y):
    VT = res_tri_mat(R, traspuesta(Q), False)
    print("listo resolviendo sistema")
    V = traspuesta(VT)
    #assert esPseudoInversa(traspuesta(Q @ R), V)
    print("listo trasponiendo")
    W = matmul(Y,V)
    
    print("Calculando W")
    return W


def pinvHouseHolder(Q, R, Y):
    """
    Usando factorizacion QR de X^T,
    Q: Matriz ortonormal de QR, calculada con HouseHolder
    R: Matriz triangular superior de QR, calculada con HouseHolder
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    return pinvQR(Q,R,Y)


def pinvGramSchmidt(Q, R, Y):
    """
    Usando factorizacion QR de X^T,
    Q: Matriz ortonormal de QR, calculada con Gram-Schmidt
    R: Matriz triangular superior de QR, calculada con Gram-Schmidt
    Y: matriz de targets de entrenamiento. 
    retorna pesos W
    """
    
    return pinvQR(Q,R,Y)

def esPseudoInversa(X, pX, tol=1e-8):
    """
    X: matrix
    pX: matrix
    retorna True si y solo si las X y xP verifican las condiciones de Moore-Penrose 
    """
    XpX = matmul(X, pX)
    pXX = matmul(pX, X)
    
    pasa_condiciones = True
    pasa_condiciones &= matricesIguales( matmul(XpX, X)  ,X,  tol )
    pasa_condiciones &= matricesIguales( matmul(pXX, pX) ,  pX , tol)
    pasa_condiciones &= esSimetrica( XpX , tol)
    pasa_condiciones &= esSimetrica( pXX , tol)
    

    return pasa_condiciones




