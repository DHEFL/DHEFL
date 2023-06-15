import warnings
import flwr as fl
import numpy as np
import pandas as pd 
from sklearn.metrics import log_loss, accuracy_score
from sklearn.svm import LinearSVC
import utils
import socket
import json
import struct
import saveCSV
import poly
import argparse
from pympler import asizeof
import config_FL
import paramsEnc
from sklearn.preprocessing import scale


if __name__ == "__main__":

    if(config_FL.sys_mode()):

        # Request params from dealer
        DEALER = "localhost"
        PORT = 10000
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((DEALER, PORT))

        def recvall(sock, n):
            #Helper function to recv n bytes or return None if EOF is hit
            data = bytearray()
            while len(data) < n:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            return data
        def recv_msg(sock):
            # Read message length and unpack it into an integer
            raw_msglen = recvall(sock, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            # Read the message data
            return recvall(sock, msglen)
        
        params = json.loads(recv_msg(client).decode())

        # Public and Secret keys
        pk1 = params['params']['pk1']
        pk2 = params['params']['pk2']
        s = params['params']['s']

        client.close()
       
        # Encryption Params
        # polynomial modulus degree
        n = paramsEnc.polynomial_modulus_degree()
        # ciphertext modulus
        q = paramsEnc.ciphertext_modulus()
        # plaintext modulus
        t = paramsEnc.plaintext_modulus()
        # polynomial modulus
        poly_mod = paramsEnc.polynomial_modulus()

    # verif partial decryption state 
    fit_state=True
  
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, config_FL.num_client()), required=True)
    args = parser.parse_args()
    indice = args.partition
    
    # Load dataset
    train_data = pd.read_csv('non-IID_MNIST/MNIST/'+str(config_FL.num_client())+'clients/client'+str(indice)+'.csv') 
    test_data = pd.read_csv('non-IID_MNIST/MNIST/scaledtest.csv')

    X_train, y_train = train_data.loc[:, 'pixel1':],train_data.loc[:, 'label']
    X_test, y_test = test_data.loc[:, 'pixel1':],test_data.loc[:, 'label']
    
    X_train = scale(X_train)
    X_test = scale(X_test)
    
    model = LinearSVC(max_iter=1)
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
       
        def get_parameters(self, config):  # type: ignore
            
            return utils.get_model_parameters(model)
        
        def fit(self, parameters, config):# type: ignore
            global indice, fit_state
            local_classes = np.asarray(np.unique(y_train), dtype='int')
            n_classes, n_features = config_FL.params_dataset()

            if (fit_state):

                # Compute Local Accuracy
                Acc_coef = parameters[0]
                Acc_intercept = parameters[1]

                for i,c in enumerate(local_classes):
                    Acc_coef[c] = model.coef_[i]
                    Acc_intercept[c] = model.intercept_[i]
                    
                utils.set_initial_params(model)
                model.coef_ = Acc_coef
                model.intercept_ = Acc_intercept
                 
                y_pred = model.predict(X_test)
                precision = accuracy_score(y_test, y_pred)

                # save local Accuracy metric
                if (config_FL.sys_mode()):
                    path_acc = "EVAL_metrics/FLEnc/local_Accuracy"+str(indice)
                    saveCSV.save(path_acc, precision, 'local Accuracy',config['server_round'],'num round')

                else:
                    path_acc = "EVAL_metrics/FL/local_Accuracy"+str(indice)
                    saveCSV.save(path_acc, precision, 'local Accuracy',config['server_round'],'num round')
                
                # update local model with global model
                global_coef = np.zeros((len(local_classes), n_features))
                global_intercept = np.zeros((len(local_classes)))
                
                for i,c in enumerate(local_classes):
                    global_coef[i] = parameters[0][c]
                    global_intercept[i] = parameters[1][c]
                    
                model.coef_ = np.round(global_coef,2)
                model.intercept_ = np.round(global_intercept,2)
                
                # Ignore convergence failure due to low local epochs
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)
                         
                local_coef = np.zeros((n_classes, n_features))
                local_intercept = np.zeros((n_classes))
                # add zeros to other classes
                for i,c in enumerate(local_classes):
                    local_coef[c] = model.coef_[i]
                    local_intercept[c] = model.intercept_[i]
                    
                local_coef = np.round(local_coef,2)    
                local_intercept = np.round(local_intercept,2)
                
                
            if(config_FL.sys_mode()):
            
                global pk1, pk2, s, n, q, t, poly_mod
                local_model = []
                 
                if fit_state :
                    
                    fit_state =False
                    
                    # convert float to integer
                    local_coef = [element * 100 for element in local_coef]
                    
                    local_intercept = [element * 100 for element in local_intercept]
                    
                    # encode the integer into a plaintext polynomial
                    for i in range(len(local_coef)):
                        local_coef[i] = np.fmod(np.array(local_coef[i].tolist() + [0] * (n - len(local_coef[i])), dtype='int64'),t)
                    
                    local_intercept = np.fmod(local_intercept + [0] * (n - len(local_intercept)),t)
                    
                    # scale coef and intercept
                    delta = q // t

                    for i in range(len(local_coef)):
                        local_coef[i] = delta * local_coef[i]
                    
                    local_intercept = delta * local_intercept

                    # generate error and binary poly for encryption
                    e1 = poly.gen_normal_poly(n)
                    e2 = poly.gen_normal_poly(n)
                    u = poly.gen_binary_poly(n)

                    # encrypt coef 
                    coef_ct0 = []
                    coef_ct1 = []

                    for i in range(len(local_coef)):
                        
                        coef_ct0.append(poly.polyadd(poly.polyadd(poly.polymul(pk1, u, q, poly_mod),e1, q, poly_mod),local_coef[i], q, poly_mod).tolist())
                        coef_ct1.append(poly.polyadd(poly.polymul(pk2, u, q, poly_mod),e2, q, poly_mod).tolist())
                    
                    coef_ct = {"coef_ct0":coef_ct0, "coef_ct1":coef_ct1}
                    coef_ct = json.dumps(coef_ct)
                    # encrypt intercept 

                    intercept_ct0 = poly.polyadd(poly.polyadd(poly.polymul(pk1, u, q, poly_mod),e1, q, poly_mod),local_intercept, q, poly_mod).tolist()
                    intercept_ct1 = poly.polyadd(poly.polymul(pk2, u, q, poly_mod),e2, q, poly_mod).tolist()
                    
                    intercept_ct = {"intercept_ct0":intercept_ct0, "intercept_ct1":intercept_ct1}
                    intercept_ct = json.dumps(intercept_ct)
                    
                    # save local data size 
                    Enc_data_client = {"fit_state":False, "coef_ct":coef_ct, "intercept_ct":intercept_ct}
                    path = 'EVAL_metrics/FLEnc/local_data_size'+str(indice+1)
                    Enc_data_size = asizeof.asizeof(Enc_data_client)
                    saveCSV.save(path, Enc_data_size, 'data size', (config['server_round']//2)+1, 'num round' )

                    return local_model, len(X_train), {"fit_state":False, "coef_ct":coef_ct, "intercept_ct":intercept_ct}
                else:
                    
                    fit_state = True
                    
                    # get global coef and intercept for decryption
                    global_coef_ct0 = parameters[0][0]
                    global_coef_ct1 = parameters[0][1]
                    
                    global_intercept_ct0 = parameters[1][0]
                    global_intercept_ct1 = parameters[1][1]

                    # generate error poly
                    error = poly.gen_normal_poly(n)

                    # partial decryption for coef

                    for i in range(len(global_coef_ct1)):
                        global_coef_ct1[i] =  poly.polyadd(poly.polymul(global_coef_ct1[i], s, q, poly_mod),error, q, poly_mod).tolist()
                    
                    global_coef_ct1 = json.dumps(global_coef_ct1.tolist())
                    global_coef_ct0 = json.dumps(global_coef_ct0.tolist())

                    # partial decryption for intercept 
                    global_intercept_ct1 = poly.polyadd(poly.polymul(global_intercept_ct1, s, q, poly_mod),error, q, poly_mod)
                    
                    global_intercept_ct1 = json.dumps(global_intercept_ct1.tolist())
                    global_intercept_ct0 = json.dumps(global_intercept_ct0.tolist())

                    local_classes = json.dumps(local_classes.tolist())

                    # save local data size
                    Dec_data_client = {"fit_state":True, "global_intercept_ct1":global_intercept_ct1, "global_coef_ct1":global_coef_ct1}                   
                    path = 'EVAL_metrics/FLEnc/local_data_size_Dec'+str(indice+1)
                    Dec_data_client_size = asizeof.asizeof(Dec_data_client)
                    saveCSV.save(path, Dec_data_client_size, 'data size', config['server_round']//2, 'num round')

                    return local_model, len(X_train), {"fit_state":True, "local_classes":local_classes, "global_intercept_ct0":global_intercept_ct0, "global_intercept_ct1":global_intercept_ct1, "global_coef_ct0":global_coef_ct0, "global_coef_ct1":global_coef_ct1}
            else:
                
                local_model = [local_coef, local_intercept]

                # save local model size
                if (config["server_round"]==1)and(indice==0):
                    size_local_model = asizeof.asizeof(local_model)
                    path = 'EVAL_metrics/FL/local_model_size'
                    saveCSV.save(path, size_local_model, 'local model size', config_FL.num_client(),'num_client')
                
                local_classes = json.dumps(local_classes.tolist())
                
                return local_model, len(X_train), {"local_classes":local_classes}
    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=MnistClient())
