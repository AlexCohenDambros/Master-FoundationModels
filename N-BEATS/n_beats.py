from darts.models import NBEATSModel
from darts import TimeSeries

from functools import partial

from sklearn.metrics import mean_absolute_percentage_error as mape
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np
import pandas as pd

import os
import time

import optuna

import multiprocessing

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

from metrics_forecasting import rrmse, pbe, pocid, mase
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler

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

def get_scaled_data(df):
    df = df[:-12]
    
    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler

def objective(trial, train_data, y_test, df_mean, scaler):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    layer_widths = trial.suggest_int("layer_widths", 16, 128, step=16)
    num_stacks = trial.suggest_int("num_stacks", 1, 3)
    num_blocks = trial.suggest_int("num_blocks", 1, 3)
    num_layers = trial.suggest_int("num_layers", 1, 4)

    train_series = TimeSeries.from_values(train_data)

    earlyStopping_nbeats = EarlyStopping(
        monitor="train_loss",
        patience=10,
        mode="min",
        min_delta=0.05,
        check_finite = True
    )

    model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=1,
        n_epochs=100,
        batch_size=batch_size,
        random_state=42,
        activation="ReLU",
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [1], "callbacks": [earlyStopping_nbeats]},
        generic_architecture=True,
        num_stacks=num_stacks,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths
    )

    model.fit(train_series)
    forecast = model.predict(12)
    y_pred = scaler.inverse_transform(forecast.values().reshape(-1, 1)).flatten()

    rrmse_result_time_moe = rrmse(y_test, y_pred, df_mean)
    print(f'\n\nRRMSE: {rrmse_result_time_moe}')
    return rrmse_result_time_moe

def create_nbeats_model(data):
    """
    Runs an N-BEATS model for time series forecasting.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing time series data.

    Returns:
    - rrmse_result (float): Relative Root Mean Squared Error.
    - mape_result (float): Mean Absolute Percentage Error.
    - pbe_result (float): Percentage Bias Error.
    - pocid_result (float): Percentage of Correct Increase or Decrease.
    - mase_result (float): Mean Absolute Scaled Error.
    - y_pred (np.ndarray): Array containing the predicted values.
    - best_params (dict): The best hyperparameters found for the model.
    """
   
    df = data['m3']
    
    df_scaled, scaler = get_scaled_data(df)

    series = TimeSeries.from_values(df_scaled)

    train_data = df_scaled[:-12]
    
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    objective_func = partial(objective, train_data=train_data, y_test=df[:-12][-12:].values, df_mean=df[:-24].mean(), scaler=scaler)
    study.optimize(objective_func, n_trials=200)
    best_params = study.best_params

    earlyStopping_nbeats = EarlyStopping(
        monitor="train_loss",
        patience=10,
        mode="min",
        min_delta=0.05,
        check_finite = True
    )

    final_model = NBEATSModel(
        input_chunk_length=12,
        output_chunk_length=1,
        n_epochs=100,
        batch_size=best_params["batch_size"],
        random_state=42,
        activation="ReLU",
        pl_trainer_kwargs={"accelerator": "gpu", "devices": [1], "callbacks": [earlyStopping_nbeats]},
        generic_architecture=True,
        num_stacks=best_params["num_stacks"],
        num_blocks=best_params["num_blocks"],
        num_layers=best_params["num_layers"],
        layer_widths=best_params["layer_widths"]
    )
    final_model.fit(series, verbose=False)
    forecast = final_model.predict(12)
    y_pred = scaler.inverse_transform(forecast.values().reshape(-1, 1)).flatten()

   # Calculating evaluation metrics
    y_baseline = df[-12*2:-12].values

    y_test = df[-12:].values

    rrmse_result_time_moe = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_time_moe = mape(y_test, y_pred)
    pbe_result_time_moe = pbe(y_test, y_pred)
    pocid_result_time_moe = pocid(y_test, y_pred)
    mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResults N-BEATS: \n")
    print(f'RRMSE: {rrmse_result_time_moe}')
    print(f'MAPE: {mape_result_time_moe}')
    print(f'PBE: {pbe_result_time_moe}')
    print(f'POCID: {pocid_result_time_moe}')
    print(f'MASE: {mase_result_time_moe}')
        
    return rrmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, best_params

def run_nbeats(state, product, data_filtered):
    """
    Execute nbeats model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the nbeats model is trained.
        - product (str): Product for which the nbeats model is trained.
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.

    Returns:
        None
    """

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        # Run nbeats model training and capture performance metrics
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_parameters = \
        create_nbeats_model(data=data_filtered)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'MODEL': 'N-BEATS',
                                    'TYPE_MODEL': 'N-BEATS',
                                    'TYPE_PREDICTIONS': np.nan,
                                    'PARAMETERS': best_parameters,
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
        
        results_df = pd.DataFrame([{'MODEL': 'N-BEATS',
                                    'TYPE_MODEL': 'N-BEATS',
                                    'TYPE_PREDICTIONS': np.nan,
                                    'PARAMETERS': np.nan,
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

    file_path = os.path.join(directory, 'nbeats_results.xlsx')
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

def run_nbeats_in_thread():

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

            # Create a separate process for running the nbeats model
            process = multiprocessing.Process(
                target=run_nbeats,
                args=(state, product, data_filtered)
            )

            # Start and wait for the process to complete
            process.start()
            process.join()

    print("All processes completed successfully.")
    
def product_and_single_thread_testing():    
    """
    Perform a simple training thread using nbeats model for time series forecasting.

    This function initializes random seeds, loads a database, executes an nbeats model,
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

    # Running the nbeats model
    y_pred = \
    create_nbeats_model(type_experiment=1, data=data_filtered_test)

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")