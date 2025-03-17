import torch
import tensorflow as tf
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import yaml
from einops import rearrange
from functools import partial
import optuna

import subprocess
import numpy as np
import pandas as pd

import os
import random
import shutil
import time
import gc

from metrics import rrmse, pbe, pocid 
from sklearn.metrics import mean_absolute_percentage_error as mape

from matplotlib import pyplot as plt

import warnings
from warnings import simplefilter

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
simplefilter(action='ignore', category=FutureWarning)
import os

os.environ["CUSTOM_DATA_PATH"] = "test_database"

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

def objective(trial, train_data_val, y_test, df_mean, type_model, type_prediction, state, derivative):
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 5e-5, 1e-6, 1e-7])
    epochs = trial.suggest_categorical("epochs", [5]) #  10, 20, 50, 80, 100
    num_layers = trial.suggest_categorical("num_layers", [1, 3, 5, 6, 8, 10])

    # Prepare data for the model
    past_target = rearrange(torch.as_tensor(train_data_val, dtype=torch.float32), "t -> 1 t 1")
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

    # INFO: ==================  FINE TUNING MODEL ==================
    folder_path = "outputs"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  
        print(f"\n=== NOTE: Delete '{folder_path}'.")
    
    print(f"\n=== Fine-tuning model {type_model} with parameters: ===")
    print(f"Fine-tuning: {type_prediction}")
    print(f"State: {state}")
    print(f"Derivative: {derivative}")
    print(f"Epoch: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"num_layers: {num_layers}")

    # INFO: ================== EDIT EPOCHS ==================
    file_path = "cli/conf/finetune/default.yaml"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['trainer']['max_epochs'] = epochs  

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

    # INFO: ================== EDIT MODEL ==================
    file_path = f"cli/conf/finetune/model/moirai_1.1_R_{type_model}.yaml"
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    data['lr'] = lr  
    data['module_kwargs']['num_layers'] = num_layers 

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)


    # INFO: ================== EDIT DATASET ==================
    file_path = "cli/conf/finetune/data/data_fine_tuning.yaml"
    with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
    
    if type_prediction == "fine_tuning_indiv":
        data['dataset'] = f'dataset_individual_{state}_{derivative}_val'
        data['storage_path'] = 'dataset_individual_val'
        
    elif type_prediction == "fine_tuning_global":
        data['dataset'] = 'dataset_global_val'
        data['storage_path'] = 'dataset_global_val'
    
    elif type_prediction == "fine_tuning_product":
        data['dataset'] = f'test_pe_etanolhidratado'
        data['storage_path'] = 'test_database'
    
    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file)

    command = [
        "python",
        "-m", "cli.train",
        "-cp", "conf/finetune",
        "run_name=fine_tuning_morai",
        f"model=moirai_1.1_R_{type_model}",
        "data=data_fine_tuning",
        "val_data=val_fine_tuning"
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
    model = MoiraiForecast(
                module = MoiraiForecast.load_from_checkpoint(checkpoint_path=f"outputs/finetune/moirai_1.1_R_{type_model}/data_fine_tuning/fine_tuning_morai", local_files_only=True),
                prediction_length=12,
                context_length=len(train_data_val),
                patch_size=16,
                num_samples=1,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
    
    # forecast
    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )
    forecast = forecast.numpy().flatten().tolist()

    y_pred = [round(value, 3) for value in forecast]
    
    gc.collect() 

    # Calculating evaluation metrics
    rrmse_result = rrmse(y_test, y_pred, df_mean) 

    print(f"\nFine-tuning results of the Morai-{type_model} model \n")
    print(f'RRMSE: {rrmse_result}')
    return rrmse_result

def create_morai_model(state, derivative, df, type_prediction='zeroshot', type_model='small'):
    """
    Creates and trains a MorAI model for time series forecasting.

    Parameters:
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - df (pd.DataFrame): The input DataFrame containing time series data.
    - type_prediction (str, optional): Specifies the prediction mode, default is 'zeroshot'.
    - type_model (str, optional): Specifies the MorAI model type. Choose from {'small', 'base'}. Default is 'small'.

    Returns:
    - rrmse_result (float): Relative Root Mean Squared Error.
    - mape_result (float): Mean Absolute Percentage Error.
    - pbe_result (float): Percentage Bias Error.
    - pocid_result (float): Percentage of Correct Increase or Decrease.
    - mase_result (float): Mean Absolute Scaled Error.
    - y_pred (np.ndarray): Array containing the predicted values.
    - best_params (dict): The best hyperparameters found for the model.
    """

    df = df['m3']

    # Data separation
    train_data = df.iloc[:-12]  
    test_data = df.iloc[-12:] 

    # Prepare training data
    train_target = train_data.to_numpy()

    # Prepare test data
    test_target = test_data.to_numpy()

    # Prepare data for the model
    past_target = rearrange(torch.as_tensor(train_target, dtype=torch.float32), "t -> 1 t 1")
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

    # INFO: ================== ZERO SHOT ==================
    if type_prediction == "zeroshot":
        # Model Initialization
        model = MoiraiForecast(
            module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{type_model}"),
            prediction_length=12,
            context_length=len(train_target),
            patch_size=16,
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    # INFO: ==================  FINE TUNING ==================
    elif type_prediction in ["fine_tuning_indiv", "fine_tuning_global", "fine_tuning_product"]:
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        objective_func = partial(objective, 
                                 train_data_val= df.iloc[:-24].to_numpy(),
                                 y_test= df[:-12][-12:].values,
                                 df_mean= df[:-24].mean(),
                                 type_model= type_model,
                                 type_prediction= type_prediction,
                                 state= state,
                                 derivative= derivative)
    
        study.optimize(objective_func, n_trials=1)
        best_params = study.best_params

        # INFO: ==================  FINE TUNING MODEL ==================
        folder_path = "outputs"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  
            print(f"\n=== NOTE: Delete '{folder_path}'.")
        
        print(f"\n=== Fine-tuning model {type_model} with parameters: ===")
        print(f"Fine-tuning: {type_prediction}")
        print(f"State: {state}")
        print(f"Derivative: {derivative}")
        print(f"Epoch: ", best_params['epochs'])
        print(f"Learning Rate: ", best_params['lr'])
        print(f"num_layers: ", best_params['num_layers'])

        # INFO: ================== EDIT EPOCHS ==================
        file_path = "cli/conf/finetune/default.yaml"
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        data['trainer']['max_epochs'] = best_params['epochs']  

        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)

        # INFO: ================== EDIT MODEL ==================
        file_path = f"cli/conf/finetune/model/moirai_1.1_R_{type_model}.yaml"
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        data['lr'] = best_params['lr']  
        data['module_kwargs']['num_layers'] = best_params['num_layers'] 

        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)


        # INFO: ================== EDIT DATASET ==================
        file_path = "cli/conf/finetune/data/data_fine_tuning.yaml"
        with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
        
        if type_prediction == "fine_tuning_indiv":
            data['dataset'] = f'dataset_individual_{state}_{derivative}'
            data['storage_path'] = 'dataset_individual'
            
        elif type_prediction == "fine_tuning_global":
            data['dataset'] = 'dataset_global'
            data['storage_path'] = 'dataset_global'
        
        elif type_prediction == "fine_tuning_product":
            data['dataset'] = f'dataset_product_{derivative}'
            data['storage_path'] = 'dataset_product'
        
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file)

        command = [
            "python",
            "-m", "cli.train",
            "-cp", "conf/finetune",
            "run_name=fine_tuning_morai",
            f"model=moirai_1.1_R_{type_model}",
            "data=data_fine_tuning"
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
        model = MoiraiForecast(
                    module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{type_model}"),
                    prediction_length=12,
                    context_length=len(train_data),
                    patch_size=16,
                    num_samples=1,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0,
                )
    
    # forecast
    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )
    forecast = forecast.numpy().flatten().tolist()

    y_pred = [round(value, 3) for value in forecast]
    
    # Display results
    print("\nForecast for the last 12 months:")
    print(y_pred)

    print("\nActual values for the last 12 months:")
    print(test_target.tolist())

    y_test = df[-12:].values

    # Calculating evaluation metrics
    y_baseline = df[-12*2:-12].values
    rrmse_result = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result = mape(y_test, y_pred)
    pbe_result = pbe(y_test, y_pred)
    pocid_result = pocid(y_test, y_pred)
    mase_result = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResults Morai_{type_model} \n")
    print(f'RRMSE: {rrmse_result}')
    print(f'MAPE: {mape_result}')
    print(f'PBE: {pbe_result}')
    print(f'POCID: {pocid_result}')
    print(f'MASE: {mase_result}')

    return rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, None
            
def run_morai(state, derivative, data_filtered, type_prediction='zeroshot', type_model='small'):

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
        create_morai_model(state=state,
                    derivative=derivative,
                    df=data_filtered, 
                    type_prediction=type_prediction, 
                    type_model=type_model)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'MODEL': 'Morai',
                                    'TYPE_MODEL': 'Morai_' + type_model,
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
        # Handle exceptions during model training
        print(f"An error occurred for product '{derivative}' in state '{state}': {e}")
        
        results_df = pd.DataFrame([{'MODEL': 'Morai',
                                    'TYPE_MODEL': 'Morai_' + type_model,
                                    'TYPE_PREDICTIONS': type_prediction,
                                    'PARAMETERS': np.nan,
                                    'STATE': state,
                                    'PRODUCT': derivative,
                                    'RRMSE': np.nan,
                                    'MAPE': np.nan,
                                    'PBE': np.nan,
                                    'POCID': np.nan,
                                    'MASE': np.nan,
                                    'PREDICTIONS': np.nan,
                                    'ERROR': f"An error occurred for product '{derivative}' in state '{state}': {e}"}])
            
    # Save the results to an Excel file if requested
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'morai_results_last_year.xlsx')
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

def run_all_in_thread(type_prediction='zeroshot', type_model='small'):

    # Load the combined dataset
    all_data = pd.read_csv('../database/combined_data.csv', sep=";")

    # Initialize a dictionary to store derivatives for each state
    state_derivative_dict = {}

    # Iterate over unique states
    for state in all_data['state'].unique():
        # Filter derivatives corresponding to this state
        derivatives = all_data[all_data['state'] == state]['product'].unique()
        # Add to the dictionary
        state_derivative_dict[state] = list(derivatives)

    # Loop through each state and its derivatives
    for state, derivatives in state_derivative_dict.items():
        for derivative in derivatives:
            print(f"========== State: {state}, derivative: {derivative} ==========")
           
            # Set random seeds for reproducibility
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            os.environ['PYTHONHASHSEED'] = str(42)
            tf.keras.utils.set_random_seed(42)

            # Filter data for the current state and derivative
            data_filtered = all_data[(all_data['state'] == state) & (all_data['product'] == derivative)]
            
            run_morai(state, derivative, data_filtered, type_prediction, type_model)

def product_and_single_thread_testing():    
    state = "pe"
    derivative = "etanolhidratado"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";", parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()
    
    rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
    create_morai_model(state=state,
                        derivative=derivative,
                        df=data_filtered_test,
                        type_prediction='fine_tuning_product',
                        type_model='small') # small, base, large
    
    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # GLOBAL: ValueError: Variate (216) exceeds maximum variate 128. 