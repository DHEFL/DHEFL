import socket, threading
import json
import poly
import numpy as np
import struct
import config_FL
from pympler import asizeof
from time import monotonic
import saveCSV
import paramsEnc

start_time = monotonic()

def getAdditiveShares(sk, M, size):
    
    #Share Generation
    shares = [poly.gen_binary_poly(size) for i in range(M-1)]
    sumshares = np.sum(shares,axis=0)
    sm = [sk[i] - sumshares[i] for i in range(len(sk))]
    sm = np.array(sm, dtype=np.int64)
    shares.append(sm)
    s=shares
    #Secret reconstruction
    secret = np.sum(s,axis=0)
    

    if ((secret == sk).all()):
        return s
    else:
        print("error secret reconstruction")

def KeyGenDealer(size, modulus, poly_mod, M):
   
    #generate secret key
    sk = poly.gen_binary_poly(size)
    s = getAdditiveShares(sk, M,size)

    #generate public key (pk1, pk2)
    a = poly.gen_uniform_poly(size, modulus)
    e = poly.gen_normal_poly(size)
  
    pk0 = poly.polyadd(poly.polymul(-a, sk, modulus, poly_mod), -e, modulus, poly_mod)
    pk1= a
   
    return pk0,pk1,s
 
class ClientThread(threading.Thread):
        def __init__(self,clientAddress,clientsocket,params):
            threading.Thread.__init__(self)
            self.csocket = clientsocket
            self.params=params
            
        def run(self):
                
            data = params.encode()
            msg = struct.pack('>I', len(data)) + data
            self.csocket.sendall(msg)
            self.csocket.close()
            
            path1 = 'EVAL_metrics/FLEnc/Params_size_Dealer'
            data_size = asizeof.asizeof(data)
            saveCSV.save(path1, data_size,'data size', config_FL.num_client(), 'nb_client')
            


LOCALHOST = "localhost"
PORT = 10000
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((LOCALHOST, PORT))

# polynomial modulus degree
n = paramsEnc.polynomial_modulus_degree()
# ciphertext modulus
q = paramsEnc.ciphertext_modulus()
# plaintext modulus
t = paramsEnc.plaintext_modulus()
# polynomial modulus
poly_mod = paramsEnc.polynomial_modulus()
# client number
global M
M = config_FL.num_client()
#public and secret keys generation
pk0, pk1 ,sk = KeyGenDealer(n, q, poly_mod,M)

i=0
while True:

        server.listen(M)
        clientsock, clientAddress = server.accept()

        params={"pk1": pk0.tolist(), "pk2": pk1.tolist()}
        
        params["s"]=sk[i].tolist()
        params = json.dumps({"params":params})

        newthread = ClientThread(clientAddress, clientsock, params)
    
        newthread.start()
        i+=1

        if(i==M):
             break


path = 'EVAL_metrics/FLEnc/runtime_Dealer'
runtime = monotonic() - start_time
saveCSV.save(path, runtime,'runtime', config_FL.num_client(), 'nb_client')
server.close()