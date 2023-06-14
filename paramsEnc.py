import numpy as np

# Encryption Params

# polynomial modulus degree
def polynomial_modulus_degree():
    n = 2**10
    return n

# ciphertext modulus
def ciphertext_modulus():
    q = 2**50 
    return q

# plaintext modulus
def plaintext_modulus():
    t = 2**16
    return t

# polynomial modulus
def polynomial_modulus():
    n = polynomial_modulus_degree()
    poly_mod = np.array([1] + [0] * (n - 1) + [1])
    return poly_mod