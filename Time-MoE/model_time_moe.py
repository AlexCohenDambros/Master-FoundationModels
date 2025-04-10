import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from functools import partial
import optuna

import subprocess

import tensorflow as tf

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import json
import shutil
import random
import time

import gc
import multiprocessing

import warnings
from warnings import simplefilter

from metrics_time_moe import rrmse, pbe, pocid 
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
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

def objective(trial, df_train, y_test, df_mean, scaler, type_model, type_prediction, state, derivative):
    lr = trial.suggest_categorical("lr", [1e-6])
    epochs = trial.suggest_categorical("epochs", [10, 20])
    global_batch_size = trial.suggest_categorical("global_batch_size", [16, 32])
    # lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 5e-5, 1e-6, 1e-7])
    # epochs = trial.suggest_categorical("epochs", [5, 10, 20, 50, 80, 100])
    # global_batch_size = trial.suggest_categorical("global_batch_size", [16, 32, 64])

    tensor_train = torch.tensor(df_train, dtype=torch.float32)
    tensor_train = tensor_train.squeeze(-1).unsqueeze(0)
    device_train = "cuda:0" if torch.cuda.is_available() else "cpu"
    tensor_train = tensor_train.to(device_train)

    # INFO: ==================  FINE TUNING MODEL ==================
    folder_path = "model_fine_tuning_optuna_last_year"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  
        print(f"\n=== NOTE: Delete '{folder_path}'.")
    
    print(f"\n=== Fine-tuning model {type_model} with parameters: ===")
    print(f"Fine-tuning: {type_prediction}")
    print(f"State: {state}")
    print(f"Derivative: {derivative}")
    print(f"Normalization: zero")
    print(f"Epoch: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"global_batch_size: {global_batch_size}")

    if type_prediction == "fine_tuning_indiv":
        dataset_path = f'dataset_individual_val/dataset_{state}_{derivative}.jsonl'

        output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
        os.makedirs(output_dir, exist_ok=True)

        command = [
            "python",
            "main.py",
            "-d", dataset_path,
            "-m", f"Maple728/TimeMoE-{type_model}",
            "-o", output_dir,
            "--num_train_epochs", str(epochs),
            "--normalization_method", 'zero',
            "--learning_rate", str(lr),
            "--global_batch_size", str(global_batch_size),
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=None)

            print("Process output:")
            print(stdout)

            if stderr:
                print("Process errors:")
                print(stderr)

        except subprocess.SubprocessError as e:
            print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
        
    elif type_prediction == "fine_tuning_global":
        dataset_path = f'dataset_global_val/dataset_global.jsonl'

        output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
        os.makedirs(output_dir, exist_ok=True)

        command = [
            "python",
            "main.py",
            "-d", dataset_path,
            "-m", f"Maple728/TimeMoE-{type_model}",
            "-o", output_dir,
            "--num_train_epochs", str(epochs),
            "--normalization_method", 'zero',
            "--learning_rate", str(lr),
            "--global_batch_size", str(global_batch_size),
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=None)

            print("Process output:")
            print(stdout)

            if stderr:
                print("Process errors:")
                print(stderr)

        except subprocess.SubprocessError as e:
            print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
    
    elif type_prediction == "fine_tuning_product":
        dataset_path = f'dataset_product_val/dataset_product_{derivative}.jsonl'

        output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
        os.makedirs(output_dir, exist_ok=True)

        command = [
            "python",
            "main.py",
            "-d", dataset_path,
            "-m", f"Maple728/TimeMoE-{type_model}",
            "-o", output_dir,
            "--num_train_epochs", str(epochs),
            "--normalization_method", 'zero',
            "--learning_rate", str(lr),
            "--global_batch_size", str(global_batch_size),
        ]

        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=None)

            print("Process output:")
            print(stdout)

            if stderr:
                print("Process errors:")
                print(stderr)

        except subprocess.SubprocessError as e:
            print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
    
    # INFO: ================== LOADING FINE TUNING MODEL ==================
    model_path =  f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
    model = TimeMoeForPrediction.from_pretrained(
        model_path,
        device_map="cuda", 
        trust_remote_code=True,
    )

    model.to(device_train)

    # forecast
    prediction_length = 12
    output = model.generate(tensor_train, max_new_tokens=prediction_length)  
    forecast = output[:, -prediction_length:] 

    forecast = forecast.cpu().numpy() 
   
    gc.collect() 

    y_pred = scaler.inverse_transform(forecast)
    y_pred = y_pred.flatten()

    # Calculating evaluation metrics
    rrmse_result = rrmse(y_test, y_pred, df_mean) 

    print(f"\nFine-tuning results of the TimeMoE-{type_model} model \n")
    print(f'RRMSE: {rrmse_result}')
    return rrmse_result

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

def create_time_moe_model(
    data, 
    state, 
    derivative, 
    type_model='50M', 
    type_prediction='zeroshot',
):
    """
    Runs a TimeMoE (Mixture of Experts for Time Series) model for forecasting.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing time series data.
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - type_model (str, optional): Specifies the TimeMoE model type. Default is '50M'.
    - type_prediction (str, optional): Specifies the prediction approach. Default is 'zeroshot'.

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

    # INFO: ================== ZERO SHOT ==================
    if type_prediction == "zeroshot":
        best_params = None
        if type_model == "50M":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-50M',
                device_map="cuda",  
                trust_remote_code=True,
            )

        elif type_model == "200M":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-200M',
                device_map="cuda",  
                trust_remote_code=True,
            )

    # INFO: ==================  FINE TUNING ==================
    elif type_prediction in ["fine_tuning_indiv", "fine_tuning_global", "fine_tuning_product"]:

        df_train = df_scaled[:-12]

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        objective_func = partial(objective, 
                                df_train= df_train,
                                y_test= df[:-12][-12:].values,
                                df_mean= df[:-24].mean(),
                                scaler= scaler,
                                type_model= type_model,
                                type_prediction= type_prediction,
                                state= state,
                                derivative= derivative)
    
        study.optimize(objective_func, n_trials=1)
        best_params = study.best_params

    tensor = torch.tensor(df_scaled, dtype=torch.float32)
    tensor = tensor.squeeze(-1).unsqueeze(0)
    device_main = "cuda:0" if torch.cuda.is_available() else "cpu"
    tensor = tensor.to(device_main)

    if type_prediction in ["fine_tuning_indiv", "fine_tuning_global", "fine_tuning_product"]:
        print(f"\n=== Fine-tuning model {type_model} with parameters: ===")
        print(f"Fine-tuning: {type_prediction}")
        print(f"State: {state}")
        print(f"Derivative: {derivative}")
        print(f"Normalization: zero")
        print(f"Epoch: ", best_params["epochs"])
        print(f"Learning Rate: ", best_params["lr"])
        print(f"global_batch_size: ", best_params["global_batch_size"])

        # INFO: ==================  FINE TUNING MODEL ==================
        folder_path = "model_fine_tuning_optuna_last_year"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  
            print(f"\n=== NOTE: Delete '{folder_path}'.")
        
        if type_prediction == "fine_tuning_indiv":
            dataset_path = f'dataset_individual/dataset_{state}_{derivative}.jsonl'

            output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
            os.makedirs(output_dir, exist_ok=True)

            command = [
                "python",
                "main.py",
                "-d", dataset_path,
                "-m", f"Maple728/TimeMoE-{type_model}",
                "-o", output_dir,
                "--num_train_epochs", str(best_params["epochs"]),
                "--normalization_method", 'zero',
                "--learning_rate", str(best_params["lr"]),
                "--global_batch_size", str(best_params["global_batch_size"]),
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=None)

                print("Process output:")
                print(stdout)

                if stderr:
                    print("Process errors:")
                    print(stderr)

            except subprocess.SubprocessError as e:
                print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
            
        elif type_prediction == "fine_tuning_global":
            dataset_path = f'dataset_global/dataset_global.jsonl'

            output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
            os.makedirs(output_dir, exist_ok=True)

            command = [
                "python",
                "main.py",
                "-d", dataset_path,
                "-m", f"Maple728/TimeMoE-{type_model}",
                "-o", output_dir,
                "--num_train_epochs", str(best_params["epochs"]),
                "--normalization_method", 'zero',
                "--learning_rate", str(best_params["lr"]),
                "--global_batch_size", str(best_params["global_batch_size"]),
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=None)

                print("Process output:")
                print(stdout)

                if stderr:
                    print("Process errors:")
                    print(stderr)

            except subprocess.SubprocessError as e:
                print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
        
        elif type_prediction == "fine_tuning_product":
            dataset_path = f'dataset_product/dataset_product_{derivative}.jsonl'

            output_dir = f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
            os.makedirs(output_dir, exist_ok=True)

            command = [
                "python",
                "main.py",
                "-d", dataset_path,
                "-m", f"Maple728/TimeMoE-{type_model}",
                "-o", output_dir,
                "--num_train_epochs", str(best_params["epochs"]),
                "--normalization_method", 'zero',
                "--learning_rate", str(best_params["lr"]),
                "--global_batch_size", str(best_params["global_batch_size"]),
            ]

            try:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(timeout=None)

                print("Process output:")
                print(stdout)

                if stderr:
                    print("Process errors:")
                    print(stderr)

            except subprocess.SubprocessError as e:
                print(f"Error executing the command for model TimeMoE-{type_model}: {e}")
            
        model_path =  f"model_fine_tuning_optuna_last_year/model_fine_tuning_{type_model}/time_moe"
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map="cuda", 
            trust_remote_code=True,
        )
    
    # INFO: ================== FORECAST ==================
    model.to(device_main)

    # forecast
    prediction_length = 12
    output = model.generate(tensor, max_new_tokens=prediction_length)  
    forecast = output[:, -prediction_length:] 

    forecast = forecast.cpu().numpy() 
   
    gc.collect() 

    y_pred = scaler.inverse_transform(forecast)
    y_pred = y_pred.flatten()

    y_test = df[-12:].values

    # Calculating evaluation metrics
    y_baseline = df[-12*2:-12].values
    rrmse_result_time_moe = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_time_moe = mape(y_test, y_pred)
    pbe_result_time_moe = pbe(y_test, y_pred)
    pocid_result_time_moe = pocid(y_test, y_pred)
    mase_result_time_moe = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResults of the TimeMoE-{type_model} model \n")
    print(f'RRMSE: {rrmse_result_time_moe}')
    print(f'MAPE: {mape_result_time_moe}')
    print(f'PBE: {pbe_result_time_moe}')
    print(f'POCID: {pocid_result_time_moe}')
    print(f'MASE: {mase_result_time_moe}')
        
    return rrmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, best_params
                    
def run_time_moe(state, derivative, data_filtered, type_prediction='zeroshot', type_model='50M'):

    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:

        # Run single TimeMoE model training
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = create_time_moe_model(
            data=data_filtered,
            state=state,
            derivative=derivative,
            type_model=type_model,
            type_prediction=type_prediction

        )

        df_all_results = pd.DataFrame([{'MODEL': 'TimeMoE',
                                        'TYPE_MODEL': 'TimeMoE_' + type_model,
                                        'TYPE_PREDICTIONS': type_prediction,
                                        'PARAMETERS': best_params,
                                        'STATE': state,
                                        'PRODUCT': derivative,
                                        'RRMSE': rrmse_result,
                                        'MAPE': mape_result,
                                        'PBE': pbe_result,
                                        'POCID': pocid_result,
                                        'MASE': mase_result,
                                        'PREDICTIONS': y_pred,
                                        'ERROR': np.nan}])

    except Exception as e:
        # Handle exceptions outside of iterations
        print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

        df_all_results = pd.DataFrame([{'MODEL': 'TimeMoE',
                                        'TYPE_MODEL': 'TimeMoE_' + type_model,
                                        'TYPE_PREDICTIONS': type_prediction,
                                        'PARAMETERS': best_params,
                                        'STATE': state,
                                        'PRODUCT': derivative,
                                        'RRMSE': rrmse_result,
                                        'MAPE': mape_result,
                                        'PBE': pbe_result,
                                        'POCID': pocid_result,
                                        'MASE': mase_result,
                                        'PREDICTIONS': y_pred,
                                        'ERROR': np.nan}])
                
    # Save results to Excel file
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'results_time_moe_last_year.xlsx')
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
    else:
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, df_all_results], ignore_index=True)
    combined_df.to_excel(file_path, index=False)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# def run_all_in_thread(type_prediction='zeroshot', type_model='50M'):
#     multiprocessing.set_start_method("spawn")

#     # Load the combined dataset
#     all_data = pd.read_csv('../database/combined_data.csv', sep=";")

#     # Initialize a dictionary to store derivatives for each state
#     state_derivative_dict = {}

#     # Iterate over unique states
#     for state in all_data['state'].unique():
#         # Filter derivatives corresponding to this state
#         derivatives = all_data[all_data['state'] == state]['product'].unique()
#         # Add to the dictionary
#         state_derivative_dict[state] = list(derivatives)

#     # Loop through each state and its derivatives
#     for state, derivatives in state_derivative_dict.items():
#         for derivative in derivatives:
#             print(f"========== State: {state}, derivative: {derivative} ==========")
           
#             # Set random seeds for reproducibility
#             random.seed(42)
#             np.random.seed(42)
#             tf.random.set_seed(42)
#             os.environ['PYTHONHASHSEED'] = str(42)
#             tf.keras.utils.set_random_seed(42)

#             # Filter data for the current state and derivative
#             data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == derivative)]
            
#             thread = multiprocessing.Process(target=run_time_moe, args=(state, derivative, data_filtered, type_prediction, type_model))
#             thread.start()
#             thread.join()  

def run_all_in_thread(type_prediction='zeroshot', type_model='50M'):
    multiprocessing.set_start_method("spawn")

    all_data = pd.read_csv('../database/combined_data.csv', sep=";")

    state_derivative_dict = {}
    for state in all_data['state'].unique():
        derivatives = all_data[all_data['state'] == state]['product'].unique()
        state_derivative_dict[state] = list(derivatives)

    results_file = 'results_model_local/results_time_moe_last_year.xlsx'

    for state, derivatives in state_derivative_dict.items():
        for derivative in derivatives:
            if os.path.exists(results_file):
                try:
                    results_df = pd.read_excel(results_file)
                    results_df.columns = [col.lower() for col in results_df.columns]
                    required_cols = ['type_model', 'type_predictions', 'state', 'product']
                    if not all(col in results_df.columns for col in required_cols):
                        print(f"Expected columns {required_cols} not found in {results_file}. Processing the combination to ensure data update.")
                    else:
                        condition = (
                            (results_df['type_model'] == f"TimeMoE_{type_model}") &
                            (results_df['type_predictions'] == type_prediction) &
                            (results_df['state'] == state) &
                            (results_df['product'] == derivative)
                        )
                        if condition.any():
                            print(f"Already processed: State = {state}, Product = {derivative}. Skipping...")
                            continue  
                except Exception as e:
                    print(f"Error reading the file {results_file}: {e}. Continuing processing.")

            print(f"========== State: {state}, Product: {derivative} ==========")

        
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == derivative)]
            
            thread = multiprocessing.Process(target=run_time_moe, args=(state, derivative, data_filtered, type_prediction, type_model))
            thread.start()
            thread.join()

def derivative_and_single_thread_testing():    
    """
    Perform a simple training thread using time_moe model for time series forecasting.

    This function initializes random seeds, loads a database, executes an time_moe model,
    evaluates its performance, and prints results.

    Parameters:
    None

    Returns:
    None
    """

    # Setting random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)
    tf.keras.utils.set_random_seed(42)

    state = "sp"
    derivative = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";",  parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the time_moe model
    rmse_result_time_moe, mape_result_time_moe, pbe_result_time_moe, pocid_result_time_moe, mase_result_time_moe, y_pred, best_params = create_time_moe_model(
                data= data_filtered_test,  
                state= state, 
                derivative= derivative, 
                type_model="50M", 
                type_prediction='fine_tuning_indiv'
     )

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")