import pandas as pd
import numpy as np
import csv
import os
import glob
from io import StringIO
import json
import matplotlib.pyplot as plt
import requests
# Modeling and Forecasting
# ==============================================================================
import sklearn
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

dire = "Test/"
base_dir = os.getcwd()
os.chdir(dire)
nomes_jsn = os.listdir(os.getcwd())
os.chdir(base_dir)

## verifando a existencia do arquivo csv
if not os.path.exists('submission.csv'):
    with open('submission.csv', 'w', newline='') as csvfile:
        colunas = ['id','mean_1','stdev_1','mean_2','stdev_2']
        writer = csv.writer(csvfile)
        writer.writerow(colunas)

rates = {}

for nome in nomes_jsn:

    with open(dire+nome) as file:

        data_json = json.load(file)
        data = pd.json_normalize(data_json)
        #data['dash']
        rate = []

        for i in range(10):
            rate.append(data['dash'][0][i].get('rate'))
        data_train = []
        for i in range(len(rate)):
            for j in rate[i]:
                data_train.append(j)
        steps = 15
        forecaster = ForecasterAutoreg(
                        regressor = RandomForestRegressor(random_state=123),
                        lags      = 12 # This value will be replaced in the grid search
                    )

        # Candidate values for lags
        lags_grid = [10, 20, 30, 40, 50, 60, 70]

        # Candidate values for regressor's hyperparameters
        param_grid = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'max_depth': [3, 8, 15, 30, 50, 70]
        }

        results_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = pd.Series(data_train),
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        metric             = 'mean_absolute_percentage_error',
                        initial_train_size = int(len(data_train)*0.5),
                        fixed_train_size   = False,
                        refit              = False,
                        skip_folds         = None,
                        return_best        = True,
                        n_jobs             = 'auto',
                        verbose            = False
                    )
        # Create and train forecaster with the best hyperparameters and lags found
        # ==============================================================================
        regressor = RandomForestRegressor(n_estimators=10, max_depth=15, random_state=123)
        forecaster = ForecasterAutoreg(
                        regressor = regressor,
                        lags      = 20
                    )
        forecaster.fit(y=pd.Series(data_train))
        # Predictions
        # ==============================================================================
        predictions = forecaster.predict(steps=steps)
        data_train2 = pd.concat([pd.Series(data_train), predictions])
        steps = 15
        forecaster = ForecasterAutoreg(
                        regressor = RandomForestRegressor(random_state=123),
                        lags      = 12 # This value will be replaced in the grid search
                    )

        # Candidate values for lags
        lags_grid = [10, 20, 30, 40, 50, 60, 70]

        # Candidate values for regressor's hyperparameters
        param_grid = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'max_depth': [3, 8, 15, 30, 50, 70]
        }

        results_grid2 = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = data_train2,
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        metric             = 'mean_absolute_percentage_error',
                        initial_train_size = int(len(data_train)*0.5),
                        fixed_train_size   = False,
                        refit              = False,
                        skip_folds         = None,
                        return_best        = True,
                        n_jobs             = 'auto',
                        verbose            = False
                    )
        # Create and train forecaster with the best hyperparameters and lags found
        # ==============================================================================
        regressor = RandomForestRegressor(n_estimators=10, max_depth=50, random_state=123)
        forecaster = ForecasterAutoreg(
                        regressor = regressor,
                        lags      = 20
                    )
        forecaster.fit(y=data_train2)
        predictions2 = forecaster.predict(steps=steps)

        np.mean(predictions2)
        np.std(predictions2)
        names = ['id','mean_1','stdev_1','mean_2','stdev_2']
        submission = pd.DataFrame(columns=names)
        row = {'id': nome[:-5],'mean_1': np.mean(predictions), 'stdev_1': np.std(predictions), 'mean_2': np.mean(predictions2), 'stdev_2': np.std(predictions2)}
        submission = submission._append(row,ignore_index = True)

        #submission
        submission.to_csv('submission.csv',header=False,mode='a', index=False)
