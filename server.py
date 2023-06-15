import flwr as fl
import utils
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import Dict
from logging import WARNING
from sklearn.svm import LinearSVC
import FedAvg_strategy
import config_FL
import pandas as pd
import numpy as np
import saveCSV
from time import monotonic
from sklearn.preprocessing import scale

start_time = monotonic()

M = config_FL.num_client()

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round,}


def get_evaluate_fn(model: LinearSVC):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    test_data = pd.read_csv('non-IID_MNIST/MNIST/scaledtest.csv')
    X_test, y_test = test_data.loc[:, 'pixel1':],test_data.loc[:, 'label']
    
    X_test = scale(X_test)
 
    #scaler = StandardScaler()
    #X_test = scaler.fit_transform(X_test)

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        if (config_FL.sys_mode()):
            if(server_round%2==0):

                utils.set_model_params(model, parameters)
                
                #loss = log_loss(y_test, model.decision_function(X_test))
                loss = []

                y_pred = model.predict(X_test)
                precision = accuracy_score(y_test, y_pred)
                
                path = 'EVAL_metrics/FLEnc/GlobalAccuracy'
                saveCSV.save(path, precision,'global Accuracy', server_round, 'num round')
                
                return loss, {"Accuracy": precision}
        else:
            utils.set_model_params(model, parameters)
            roundTime = monotonic() - start_time
            pathRT = 'EVAL_metrics/FL/roundTime'
            saveCSV.save(pathRT, roundTime,'Time', server_round, 'num round')

            loss = []
            # Predict the labels of the test set
            y_pred = model.predict(X_test)
            precision = accuracy_score(y_test, y_pred)
            
            path = 'EVAL_metrics/FL/GlobalAccuracy'
            saveCSV.save(path, precision,'global Accuracy', server_round, 'num round')
  
            return loss, {"Accuracy": precision}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LinearSVC()
    utils.set_initial_params(model)

    strategy = FedAvg_strategy.FedAvg(
            fraction_fit=1.0,
            min_fit_clients=M,
            min_available_clients=M,
            evaluate_fn=get_evaluate_fn(model),
            on_fit_config_fn=fit_round,
    )
    
    
    if(config_FL.sys_mode()):
        nb_round = 2*config_FL.nb_round()
    else:
        nb_round = config_FL.nb_round()

    history = fl.server.start_server(
        server_address = "[::]:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=nb_round),
    )
    
    runtime = monotonic() - start_time
    
    if config_FL.sys_mode():
        path = 'EVAL_metrics/FLEnc/runtime'
    else:
        path = 'EVAL_metrics/FL/runtime'

    saveCSV.save(path, runtime,'Time', M, 'num client')
    
    
    """
    acc_val=[]
    acc = history.metrics_centralized["Accuracy"]

    for i in range(len(acc)):
        acc_val.append(acc[i][1])
   """ 

    