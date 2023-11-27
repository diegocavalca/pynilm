import os
import sys
import time
import json
import types
import itertools
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.base import clone
from collections import defaultdict
from IPython.display import clear_output

from pynilm.data import DataWrapper
from pynilm.metrics import *

class Experiment:    
    def __init__(
        self,
        models,
        metrics,
        train_datasource,
        test_datasource,
        name=None,
        model_type='binary',
    ):
        """
        Perform experiments on NILM context.
        
        Args:
            models (sklearn.pipeline.Pipeline): One or many ML model to evaluate.
            metrics (sklearn.metrics): Model evaluation metric functions.
            
        """
        
        self.name = name
        if self.name == None:
            self.name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.models = models
        
        # self.mains_label = mains_label
        
        self.train_datasource = train_datasource        
        if isinstance(train_datasource, DataWrapper):
            self.train_mains = train_datasource.mains
        elif isinstance(train_datasource, list):
            self.train_mains = np.concatenate([ds.mains for ds in self.train_datasource])
        else:
            raise Exception('Invalid `train_datasource` type (DataWrapper or list of DataWrappers only)')
        # self.train_data = np.array([data[self.mains_label].values for data in self.train_data])
            
        self.train_activations = defaultdict(list)
        for ds in self.train_datasource:
            for key, values in ds.activations.items():
                self.train_activations[key].extend(values)
        self.train_activations = dict(self.train_activations)
        
        self.test_datasource = test_datasource
        if isinstance(test_datasource, DataWrapper):
            self.test_mains = self.test_datasource.mains
        elif isinstance(test_datasource, list):
            self.test_mains = np.concatenate([ds.mains for ds in self.test_datasource])
        else:
            raise Exception('Invalid `test_datasource` type (DataWrapper or list of DataWrappers only)')
        # self.test_data = np.array([data[self.mains_label].values for data in self.test_data])
        
        self.test_activations = defaultdict(list)
        for ds in self.test_datasource:
            for key, values in ds.activations.items():
                self.test_activations[key].extend(values)
        self.test_activations = dict(self.test_activations)
        
        self.metrics = metrics
        if self.metrics == None or isinstance(self.metrics, list) or \
            (isinstance(self.metrics, dict) and len(self.metrics) == 0):
            raise Exception('Please provide at least one valid metric')
            
        self.model_type = model_type
        
    def run(self, n_jobs=-1, get_params=True):
        
        results = []
        
        X_train = self.train_mains
        X_test = self.test_mains
        
        for model_name, model_pipeline in self.models.items():
            clear_output()
            print(f'Analyzing `{model_name}` pipeline')

            # TODO: refactor different actions in methods (data preparation,
            #   model fitting, performance evaluation, etc.
            if self.model_type == 'binary':

                # if model_name not in results.keys():
                #     results[model_name] = {}

                for appliance in self.train_activations.keys():

                    # if appliance not in results[model_name].keys():
                    #     results[model_name][appliance] = {}

                    result = {
                        'model': model_name,
                        'appliance': appliance,
                    }

                    # Preparing data
                    y_train = np.array(self.train_activations[appliance])
                    y_test = np.array(self.test_activations[appliance])

                    # Fitting the model
                    # start = time.time()
                    # model = clone(model_pipeline)
                    # model.fit(X_train, y_train)
                    # training_time = time.time() - start
                    print(f'Fitting {appliance} model')
                    model, training_time = self.__fit__(model_pipeline, X_train, y_train)

                    # Prformance evaluation on test set
                    result_eval, pred_time = self.__evaluate__(model, X_test, y_test)
                    result.update(result_eval)

                    # Additional information
                    result['training_time'] = training_time
                    result['prediction_time'] = pred_time
                    if get_params:
                        result['model_pipeline_params'] = self.__get_params__(model)

                    # TODO: persist results and models

                    results.append(result)

            elif self.model_type == 'multilabel':

                result = {'model': model_name}

                # Preparing data
                y_train = pd.DataFrame(self.train_activations).values
                y_test = pd.DataFrame(self.test_activations).values

                # Fitting the model
                # start = time.time()
                # model = clone(model_pipeline)
                # model.fit(X_train, y_train)
                # training_time = time.time() - start
                model, training_time = self.__fit__(model_pipeline, X_train, y_train)

                # Prformance evaluation on test set
                result_eval, pred_time = self.__evaluate__(model, X_test, y_test)
                result.update(result_eval)

                # Additional information
                result['training_time'] = training_time
                result['prediction_time'] = pred_time
                if get_params:
                    result['model_pipeline_params'] = self.__get_params__(model)

                results.append(result)

            else:

                raise Exception(f'Invalid `model_type` ({self.model_type}).')
                    
        return results#pd.read_json(json.dumps(results), orient='records')
    
    def __fit__(self, model, X_train, y_train):
        start = time.time()
        clf = clone(model)
        # tf.keras.models.clone_model
        clf.fit(X_train, y_train)
        training_time = time.time() - start
        
        return clf, training_time
        
    
    def __evaluate__(self, model, X_test, y_test):
        result = {}
        # Prformance evaluation on test set
        total_pred_time = 0                    
        for metric_name, metric_method in self.metrics.items():
            print(f'Evaluating {metric_name}')

            score, pred_time = self.__eval_metric__(metric_method, model, X_test, y_test)

            result[metric_name] = score
            total_pred_time += pred_time
            
        mean_pred_time = total_pred_time / len(self.metrics)
        
        return result, mean_pred_time
        
    def __eval_metric__(self, metric_method, clf, X_test, y_test):
        
        # results[model_name][appliance][metric_name] = metric_method(y_test, y_pred)
        # Simple Python method / sklearn raw metric
        if isinstance(metric_method, str):
            
            # Generating predictions
            start = time.time()
            y_pred = clf.predict(X_test)
            pred_time = time.time() - start
            
            metric_method = globals()[metric_method] 
            
            score = metric_method(y_test, y_pred)
        
        elif isinstance(metric_method, types.FunctionType):
            
            # Generating predictions
            start = time.time()
            y_pred = clf.predict(X_test)
            pred_time = time.time() - start

            score = metric_method(y_test, y_pred)

        # if is a custom scorer (sklearn)
        elif getattr(metric_method, '__module__', None) == 'sklearn.metrics._scorer':

            # Generating predictions
            start = time.time()
            score = metric_method(clf, X_test, y_test)
            pred_time = time.time() - start

        else:
            score = None
            pred_time = 0
        
        return score, pred_time
        
    
    def __get_params__(self, model_pipeline):
        return {p: v for p, v in model_pipeline.get_params().items() if '__' in p}
        