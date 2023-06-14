import numpy as np
from numpy.polynomial import polynomial as poly


def gen_binary_poly(size):
    
    return np.random.randint(-1, 2, size, dtype=np.int64)

def gen_uniform_poly(size, modulus):
    
    return np.random.randint(0, modulus, size, dtype=np.int64)

def gen_normal_poly(size):
    
    return np.int64(np.random.normal(0, 2, size=size))

def polymul(x, y, modulus, poly_mod):
    
    return np.int64(np.round(np.fmod(poly.polydiv(np.fmod(poly.polymul(x, y),modulus),poly_mod)[1], modulus)))

def polyadd(x, y, modulus, poly_mod):
   
    return np.int64(np.round(np.fmod(poly.polydiv(np.fmod(poly.polyadd(x, y),modulus),poly_mod)[1], modulus)))

