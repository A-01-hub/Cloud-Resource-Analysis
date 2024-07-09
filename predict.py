import logging
import pickle
import pandas as pd

"""
This file should be named as 'predict.py' and should contain the 'ModelServe' class and the 'predict' method.
"""
class ModelServe:

    def __init__(self):
        """
        Initialization method for the deployment. Invoked once during deployment startup.
        Load your ML models here and use them in the predict function for serving individual requests.
        """
        logging.info('Initializing models for serving prediction requests')

    def predict(self, request):
        """
        Return model prediction for a request. Invoked for every individual request.
        Implement this method.

        Arguments:
        request -- a Python dictionary representing JSON body of a prediction request.
        """
        logging.info('Processing the prediction request')
        model_loaded1=pickle.load(open('KNN_model','rb'))
        model_loaded2=pickle.load(open('KNN_model2','rb'))

        a=model_loaded1.predict(pd.DataFrame([[1.590251,-0.457010,1.237621,0.124644,0.827392]],columns=['network_traffic','power_consumption','num_executed_instructions','execution_time','energy_efficiency']))
        b=model_loaded2.predict(pd.DataFrame([[1.590251,-0.457010,1.237621,0.124644,0.827392]],columns=['network_traffic','power_consumption','num_executed_instructions','execution_time','energy_efficiency']))

        model_prediction = {'CPU':0,'memory_usage':0}
        std_deviation_of_cpu=28.489758101240163
        mean_of_cpu=46.10329593291688
        a=(a*std_deviation_of_cpu)+mean_of_cpu
        std_deviation_of_memory=29.196124181190324
        mean_of_memory=49.810549649547
        b=(a*std_deviation_of_memory)+mean_of_memory
        model_prediction['CPU']=a
        model_prediction['memory_usage']=b
        return model_prediction

model =ModelServe()
print(model.predict())