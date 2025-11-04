#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:32:39 2025

@author: Estudiante
"""
 


# esta m al



import numpy as np
from labo00_auxiliares import *


a=np.cos(0)
pi=np.pi

def rota(theta):
        return np.array([
            [np.cos(theta),-np.sin(theta)],
            [np.sin(theta),np.cos(theta)]])
    
    
    
def escala(vector):
    res=np.zeros((len(vector),len(vector)))
    for i in range(len(vector)):
        res[i][i] = vector[i]
    return res

escala([1,2,3,34,5,6])


def rota_y_escala(theta,s):
    res = matmul(escala(s), rota(theta))
    return res
    
rota_y_escala(pi/2,[1,2])
 
def afin(theta,s,b):
    res= rota_y_escala(theta, s)
    res = np.append(res,[[0,0]],0)
    res =  np.append(res,[[b[0]],[b[1]],[1]],1)
    return res

xd=afin(pi/4,[1,2],[1,2])

def trans_afin(v,theta,s,b):
    mult = v + [1]
    return calcularAx(afin(theta,s,b), mult)

trans_afin([1,0],pi/4,[1,2],[1,2])

    
    
    

    
