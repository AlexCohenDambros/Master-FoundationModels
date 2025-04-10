import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from functools import partial
import optuna

from darts.models import RNNModel
from darts import TimeSeries

import torch
import torch.nn as nn
import torch.optim as optim
from darts.metrics import mse

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import time

import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from metrics_lstm import rrmse, pbe, pocid, mase
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

from functions_forecasting import recursive_multistep_forecasting

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
simplefilter(action='ignore', category=FutureWarning)

#Reproducibilty
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'


# Function to convert a date string into a pandas Timestamp
def convert_date(date_string):
    """
    Convert a date string into a pandas Timestamp.

    Parameters:
    - date_string: str, date in 'YYYYMM' format

    Returns:
    - pd.Timestamp object representing the date
    """
    year_month = date_string.strip()
    year = int(year_month[:4])
    month = int(year_month[4:])
    return pd.Timestamp(year=year, month=month, day=1)

def minmax_scaler(df):
    df = df[:-12]
    
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def objective(trial, train_data_val, y_test, df_mean, scaler):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, step=32)
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 3)
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.10])
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 5e-5, 2e-5, 1e-6])
    n_epochs = trial.suggest_categorical("n_epochs", [10, 20, 50, 80, 100])

    model = RNNModel(
        model="LSTM",
        input_chunk_length=12,
        n_epochs=n_epochs,
        batch_size=16,
        random_state=42,
        hidden_dim=hidden_dim,             # Dimensão oculta (exemplo: 64)
        n_rnn_layers=n_rnn_layers,            # Número de camadas LSTM (exemplo: 2)
        dropout=dropout,               # Taxa de dropout (exemplo: 10%)
        optimizer_cls=optim.Adam,  # Otimizador utilizado (Adam)
        loss_fn=nn.MSELoss(),      # Função de perda (MSE Loss)
        optimizer_kwargs={"lr": lr},
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [1]},
    )

    model.fit(train_data_val)
    forecast = model.predict(12)
    y_pred = scaler.inverse_transform(forecast.values().reshape(-1, 1)).flatten()
 
    rrmse_result_time_moe = rrmse(y_test, y_pred, df_mean)
    print(f'\n\nRRMSE: {rrmse_result_time_moe}')
    return rrmse_result_time_moe

def create_lstm_model(data):
    """
    Creates and trains an LSTM model for time series forecasting using recursive or direct predictions.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing time series values.

    Returns:
    - rrmse_result_lstm (float): Relative Root Mean Squared Error.
    - mape_result_lstm (float): Mean Absolute Percentage Error.
    - pbe_result_lstm (float): Percentage Bias Error.
    - pocid_result_lstm (float): Percentage of Correct Increase or Decrease.
    - mase_result_lstm (float): Mean Absolute Scaled Error.
    - y_pred (np.ndarray): Array containing the predicted values.
    - best_params (dict): The best hyperparameters found for the model.
    """

    df = data['m3']
    
    df, scaler = minmax_scaler(df)
    df_val, scaler_val = minmax_scaler(df[:-12])

    series = TimeSeries.from_values(df)
    series_val = TimeSeries.from_values(df_val)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    objective_func = partial(objective, train_data_val=series_val, y_test=df[:-12][-12:].values, df_mean=df[:-24].mean(), scaler=scaler_val)
    study.optimize(objective_func, n_trials=200)
    best_params = study.best_params
    
    # ======== LSTM ========
    model_lstm = RNNModel(
        model="LSTM",
        input_chunk_length=12,
        n_epochs=best_params["n_epochs"],
        batch_size=16,
        random_state=42,
        hidden_dim=best_params["hidden_dim"],             # Dimensão oculta (exemplo: 64)
        n_rnn_layers=best_params["n_rnn_layers"],            # Número de camadas LSTM (exemplo: 2)
        dropout=best_params["dropout"],               # Taxa de dropout (exemplo: 10%)
        optimizer_cls=optim.Adam,  # Otimizador utilizado (Adam)
        loss_fn=nn.MSELoss(),      # Função de perda (MSE Loss)
        optimizer_kwargs={"lr": best_params["lr"]},
    )

    # Treinamento do modelo
    model_lstm.fit(series, verbose=True)

    # Predição dos próximos valores da série
    predictions = model_lstm.predict(12)

    # Como os valores preditos estão normalizados, é preciso
    # transformá-los para a escala original em M3

    # Reescala as predições
    y_pred = scaler.inverse_transform(predictions.values().reshape(-1, 1)).flatten()

    # Recupera os valores reais na escala original
    y_test = df[-12:].values
    y_baseline = df[-12 * 2:-12].values

    # Evaluation metrics
    y_baseline = df[-12*2:-12].values
    rrmse_result_lstm = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_lstm = mape(y_test, y_pred)
    pbe_result_lstm = pbe(y_test, y_pred)
    pocid_result_lstm = pocid(y_test, y_pred)
    mase_result_lstm = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nMetrics: LSTM \n")
    print(f'RRMSE: {rrmse_result_lstm}')
    print(f'MAPE: {mape_result_lstm}')
    print(f'PBE: {pbe_result_lstm}')
    print(f'POCID: {pocid_result_lstm}')
    print(f'MASE: {mase_result_lstm}')
        
    return rrmse_result_lstm, mape_result_lstm, pbe_result_lstm, pocid_result_lstm, mase_result_lstm, y_pred, best_params
                    
def run_lstm(state, product, data_filtered):
    """
    Execute LSTM model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the LSTM model is trained.
        - product (str): Product for which the LSTM model is trained.
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.

    Returns:
        None
    """

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        # Run LSTM model training and capture performance metrics
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
        create_lstm_model(data=data_filtered)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'MODEL': 'LSTM',
                                    'TYPE_MODEL': 'LSTM',
                                    'TYPE_PREDICTIONS': 'LSTM',
                                    'PARAMETERS': best_params,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': rrmse_result,
                                    'MAPE': mape_result,
                                    'PBE': pbe_result,
                                    'POCID': pocid_result,
                                    'MASE': mase_result,
                                    'PREDICTIONS': y_pred,
                                    'ERROR': np.nan}])
    except Exception as e:
        # Handle exceptions during model training
        print(f"An error occurred for product '{product}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'MODEL': 'LSTM',
                                    'TYPE_MODEL': 'LSTM',
                                    'TYPE_PREDICTIONS': 'LSTM',
                                    'PARAMETERS': best_params,
                                    'STATE': state,
                                    'PRODUCT': product,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{product}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'lstm_results.xlsx')
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
    else:
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    combined_df.to_excel(file_path, index=False)

    ## Calculate and display the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def run_lstm_in_thread():
    """
    Execute LSTM model training in separate processes for different state and product combinations.

    Returns:
        None
    """
    # Set the multiprocessing start method
    multiprocessing.set_start_method("spawn")

     # Load combined dataset
    data_path = '../database/combined_data.csv'
    try:
        all_data = pd.read_csv(data_path, sep=";")
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # Group products by state for efficient processing
    state_product_dict = {
        state: list(all_data[all_data['state'] == state]['product'].unique())
        for state in all_data['state'].unique()
    }

    # Iterate through each state and associated products
    for state, products in state_product_dict.items():
        for product in products:
            print(f"========== Processing State: {state}, Product: {product} ==========")

            # Filter data for the current state and product
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == product)]

            # Create a separate process for running the LSTM model
            process = multiprocessing.Process(
                target=run_lstm,
                args=(
                    state, product,
                    data_filtered
                )
            )

            # Start and wait for the process to complete
            process.start()
            process.join()

    print("All processes completed successfully.")
    
def product_and_single_thread_testing():    
    """
    Perform a simple training thread using LSTM model for time series forecasting.

    This function initializes random seeds, loads a database, executes an LSTM model,
    evaluates its performance, and prints results.

    Parameters:
        None

    Returns:
        None
    """

    state = "sp"
    product = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{product}/mensal_{state}_{product}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {product} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the LSTM model
    rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
    create_lstm_model(data=data_filtered_test)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")