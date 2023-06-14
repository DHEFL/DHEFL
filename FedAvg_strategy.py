from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy
import json
import poly
import numpy as np
import config_FL
from pympler import asizeof
import saveCSV
import paramsEnc

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

# flake8: noqa: E501
class FedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        nb_client= config_FL.num_client()
        n_classes, n_features = config_FL.params_dataset()

        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        if(config_FL.sys_mode()):
            

            # polynomial modulus degree
            n = paramsEnc.polynomial_modulus_degree()
            # ciphertext modulus
            q = paramsEnc.ciphertext_modulus()
            # plaintext modulus
            t = paramsEnc.plaintext_modulus()
            # polynomial modulus
            poly_mod = paramsEnc.polynomial_modulus()

            if fit_metrics[0][1]['fit_state']:

                global_dec_coef = np.zeros((n_classes,n), dtype=np.int64)
                global_dec_intercept = np.zeros((n), dtype=np.int64)

                global_coef_ct0 = json.loads(fit_metrics[0][1]['global_coef_ct0'])        
                global_intercept_ct0 = json.loads(fit_metrics[0][1]['global_intercept_ct0'])

                class_counts = np.zeros((n_classes))


                # sum partial decryption for coef and intercept
                for i in range(nb_client):

                    # get partial dec coef and intercept from client i
                    dec_coef = json.loads(fit_metrics[i][1]['global_coef_ct1'])
                    dec_intercept = json.loads(fit_metrics[i][1]['global_intercept_ct1'])

                    local_classes = json.loads(fit_metrics[i][1]['local_classes'])
                    
                    for class_num in range(n_classes):
                        if class_num in local_classes:
                            class_counts[class_num] += 1

                    for j in range(n_classes):
                    
                        global_dec_coef[j] = poly.polyadd(global_dec_coef[j], dec_coef[j], q, poly_mod)
                    
                    global_dec_intercept = poly.polyadd(global_dec_intercept, dec_intercept, q, poly_mod)
                
               
                # final decryption for coef
                F_global_dec_coef = np.zeros((n_classes,n_features))
                for i in range(n_classes):
                    g_coef = poly.polyadd(global_coef_ct0[i], global_dec_coef[i], q, poly_mod)
                    new_gcoef = np.fmod(np.round(t * g_coef / q),t)
                    avg_coef = [element / class_counts[i] for element in new_gcoef]
                    new_avg_coef = [element / 100 for element in avg_coef]
                    F_global_dec_coef[i] = new_avg_coef[0:n_features]
                

                # final decryption for intercept 
                F_global_dec_intercept = np.zeros((n_classes))
                g_intercept = poly.polyadd(global_intercept_ct0, global_dec_intercept, q, poly_mod)
                new_gintercept = np.fmod(np.round(t * g_intercept / q),t)
                avg_intercept = [element / nb for element, nb in zip(new_gintercept,class_counts)]
                new_avg_intercept = [element / 100 for element in avg_intercept]
                F_global_dec_intercept = new_avg_intercept[0:n_classes]
                
                F_global_dec_coef = np.round(F_global_dec_coef,2)
                F_global_dec_intercept = np.round(F_global_dec_intercept,2)
                parameters_aggregated = ndarrays_to_parameters ([F_global_dec_coef, F_global_dec_intercept])

                # save global model size
                if server_round == 2:
                    
                    path = 'EVAL_metrics/FLEnc/Global_model_dec'
                    Global_model_dec = asizeof.asizeof(parameters_aggregated)
                    saveCSV.save(path, Global_model_dec, 'Global model dec', server_round, 'num round')
                
                metrics_aggregated = {}
   
                return parameters_aggregated, metrics_aggregated
                
               
            else:

                # initialize global_coef and intercept for aggregation

                global_coef_ct0 = np.zeros((n_classes,n),dtype=np.int64)
                global_coef_ct1 = np.zeros((n_classes,n),dtype=np.int64)

                global_intercept_ct0 = np.zeros((n),dtype=np.int64)
                global_intercept_ct1 = np.zeros((n),dtype=np.int64)

                for i in range(nb_client):
                    
                    # get local coef of client i
                    local_coef = json.loads(fit_metrics[i][1]['coef_ct'])
                    local_coef_ct0 = local_coef['coef_ct0']
                    local_coef_ct1 = local_coef['coef_ct1']

                    # get local intercept of client i
                    local_intercept = json.loads(fit_metrics[i][1]['intercept_ct'])
                    local_intercept_ct0 = local_intercept['intercept_ct0']
                    local_intercept_ct1 = local_intercept['intercept_ct1']

                    for j in range(n_classes):
                        global_coef_ct0[j] = poly.polyadd(global_coef_ct0[j], local_coef_ct0[j], q, poly_mod)
                        global_coef_ct1[j] = poly.polyadd(global_coef_ct1[j], local_coef_ct1[j], q, poly_mod)
                    
                    global_intercept_ct0 = poly.polyadd(global_intercept_ct0, local_intercept_ct0, q, poly_mod)
                    global_intercept_ct1 = poly.polyadd(global_intercept_ct1, local_intercept_ct1, q, poly_mod)
                
                global_coef_ct = [global_coef_ct0, global_coef_ct1]
                global_intercept_ct = [global_intercept_ct0, global_intercept_ct1]

                parameters_aggregated = ndarrays_to_parameters ([global_coef_ct, global_intercept_ct])
                
                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                
                # save global model size
                if server_round == 1:
                    path = 'EVAL_metrics/FLEnc/Global_model_Enc'
                    Global_model_Enc = asizeof.asizeof([global_coef_ct1, global_intercept_ct1])
                    saveCSV.save(path, Global_model_Enc, 'Global_model_Enc', server_round, 'num round')
                   
                return parameters_aggregated, metrics_aggregated     
        else:
            
            global_coef = np.zeros((n_classes, n_features))
            global_intercept = np.zeros((n_classes))
            class_counts = np.zeros((n_classes))

            weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results]
                       
            for i in range(nb_client):
                
                local_coef = weights_results[i][0][0]
                local_intercept = weights_results[i][0][1]

                local_classes = json.loads(fit_metrics[i][1]['local_classes'])
                    
                for class_num in range(n_classes):
                    if class_num in local_classes:
                        class_counts[class_num] += 1
                
                for j in range(len(global_coef)):
                    global_coef[j]= [var1 + var2 for var1, var2 in zip(global_coef[j],local_coef[j])]
               
                global_intercept= [var1 + var2 for var1, var2 in zip(global_intercept,local_intercept)]

               
            
            for i in range(len(global_coef)):
                global_coef[i] = [element / class_counts[i] for element in global_coef[i]]
                   
            global_intercept = [element / nb for element,nb in zip(global_intercept,class_counts)]

            global_coef = np.round(global_coef,2)
            global_intercept = np.round(global_intercept,2)
            
            parameters_aggregated = ndarrays_to_parameters ([global_coef,global_intercept])

            if (server_round==1):
                size_global_model = asizeof.asizeof(parameters_aggregated)
                path = 'EVAL_metrics/FL/global_model_size'
                saveCSV.save(path, size_global_model, 'global model size', nb_client,'num_client')
            
            #parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
            metrics_aggregated = {}
            if self.fit_metrics_aggregation_fn:
                fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")

            return parameters_aggregated, metrics_aggregated
   

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
