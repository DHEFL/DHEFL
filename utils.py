from typing import Tuple, Union, List
import numpy as np
from sklearn.svm import LinearSVC
import config_FL


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LinearSVC) -> LogRegParams:
    """Returns the paramters of a sklearn SVM model."""
    if model.fit_intercept:
        
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        
        params = [
            model.coef_,
        ]
    params[0]=np.round(params[0],2)
    params[1]=np.round(params[1],2)
    return params

def set_model_params(
    model: LinearSVC, params: LogRegParams
) -> LinearSVC:
    
    """Sets the parameters of a sklean SVM model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LinearSVC):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.SVM documentation for more
    information.
    """
    n_classes,n_features = config_FL.params_dataset()

    model.classes_ = np.array([i for i in range(n_classes)])

    model.coef_ = np.zeros((n_classes, n_features))
    
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes))