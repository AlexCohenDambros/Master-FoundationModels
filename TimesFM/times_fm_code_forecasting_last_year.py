import tensorflow as tf
import timesfm
from src.finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder

import numpy as np
import pandas as pd

from functools import partial
import optuna

import gc

import os
import random
import time

import warnings
from warnings import simplefilter

from metrics_times import rrmse, pbe, pocid 
from sklearn.metrics import mean_absolute_percentage_error as mape

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


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

def objective(trial, df_train, y_test, df_mean, type_model, state, derivative):
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 5e-5, 1e-6, 1e-7])
    epochs = trial.suggest_categorical("epochs", [5, 10, 20, 50, 80, 100])
    global_batch_size = trial.suggest_categorical("global_batch_size", [16, 32, 64])

    # INFO: ==================  FINE TUNING MODEL ==================
    if type_model == '200M':
        model_tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=16,
                horizon_len=12,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )
    
    else:
        model_tfm = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=16,                
                horizon_len=12,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )

    """Basic example of finetuning TimesFM on stock data."""
    model, hparams, tfm_config = get_model(load_weights=True)
    config = FinetuningConfig(batch_size=256,
                                num_epochs=5,
                                learning_rate=1e-4,
                                use_wandb=False,
                                freq_type=1,
                                log_every_n_steps=10,
                                val_check_interval=0.5,
                                use_quantile_loss=True)

    train_dataset, val_dataset = get_data(128,
                                            tfm_config.horizon_len,
                                            freq_type=config.freq_type)
    finetuner = TimesFMFinetuner(model, config)

    print("\nStarting finetuning...")
    results = finetuner.finetune(train_dataset=train_dataset,
                                val_dataset=val_dataset)

    print("\nFinetuning completed!")
    print(f"Training history: {len(results['history']['train_loss'])} epochs")

    # forecast
    prediction_length = 12
    output = model.generate(tensor_train, max_new_tokens=prediction_length)  
    forecast = output[:, -prediction_length:] 

    forecast = forecast.cpu().numpy() 
   
    gc.collect() 

    y_pred = forecast.flatten()

    # Calculating evaluation metrics
    rrmse_result = rrmse(y_test, y_pred, df_mean) 

    print(f"\nFine-tuning results of the TimeMoE-{type_model} model \n")
    print(f'RRMSE: {rrmse_result}')
    return rrmse_result

def create_times_fm(
    state,
    derivative,
    df, 
    type_prediction='zeroshot',
    type_model='200M',
):
    """
    Runs a TimeFM model for time series forecasting.

    Parameters:
    - state (str): The specific state for which the model is trained.
    - derivative (str): The specific derivative for which the model is trained.
    - df (pd.DataFrame): The input DataFrame containing time series data.
    - type_prediction (str, optional): Specifies the prediction approach. Default is 'zeroshot'.
    - type_model (str, optional): Specifies the TimeFM model type. Choose from {'200M', '500M'}. Default is '200M'.

    Returns:
    - rrmse_result (float): Relative Root Mean Squared Error.
    - mape_result (float): Mean Absolute Percentage Error.
    - pbe_result (float): Percentage Bias Error.
    - pocid_result (float): Percentage of Correct Increase or Decrease.
    - mase_result (float): Mean Absolute Scaled Error.
    - y_pred (np.ndarray): Array containing the predicted values.
    - best_params (dict): The best hyperparameters found for the model.
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    df['unique_id'] = 1
    df.rename(columns={'timestamp': 'ds'}, inplace=True)
    df = df[['unique_id', 'ds', 'm3']] 
    
    train_data = df.iloc[:-12]  
    test_data = df.iloc[-12:]  

    # INFO: ================== ZERO SHOT ==================
    if type_prediction == "zeroshot":
        best_params = None
        if type_model == '200M':
            model_tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=16,
                    horizon_len=12,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
            )
        
        else:
            model_tfm = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu",
                    per_core_batch_size=16,                
                    horizon_len=12,
                    num_layers=50,
                    use_positional_embedding=False,
                    context_len=2048,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
            )

    # INFO: ==================  FINE TUNING ==================
    elif type_prediction in ["fine_tuning_indiv", "fine_tuning_global", "fine_tuning_product"]:

        df_train = train_data[:-12]

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
        objective_func = partial(objective, 
                                df_train= df_train,
                                y_test= df["m3"][:-12][-12:].values,
                                df_mean= df["m3"][:-24].mean(),
                                type_model= type_model,
                                type_prediction= type_prediction,
                                state= state,
                                derivative= derivative)
    
        study.optimize(objective_func, n_trials=100)
        best_params = study.best_params
    
    if type_prediction in ["fine_tuning_indiv", "fine_tuning_global", "fine_tuning_product"]:
        print(f"\n=== Fine-tuning model {type_model} with parameters: ===")
        print(f"Fine-tuning: {type_prediction}")
        print(f"State: {state}")
        print(f"Derivative: {derivative}")

        # INFO: ==================  FINE TUNING MODEL ==================

    # forecast
    forecast_df = model_tfm.forecast_on_df(
            inputs=train_data,
            freq="M",  # monthly
            value_name="m3",
            num_jobs=-1,
        )

    y_pred = forecast_df["timesfm"].tolist()

    y_pred = [round(value, 3) for value in y_pred]

    # Display results
    print("\nForecast for the last 12 months:")
    print(y_pred)

    print("\nActual values for the last 12 months:")
    print(test_data["m3"].tolist())

    y_test = df["m3"][-12:].values

    # Calculating evaluation metrics
    y_baseline = df["m3"][-12*2:-12].values
    rrmse_result = rrmse(y_test, y_pred, df["m3"][:-12].mean())
    mape_result = mape(y_test, y_pred)
    pbe_result = pbe(y_test, y_pred)
    pocid_result = pocid(y_test, y_pred)
    mase_result = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nResultados TimesFM modelo: {type_model} \n")
    print(f'RRMSE: {rrmse_result}')
    print(f'MAPE: {mape_result}')
    print(f'PBE: {pbe_result}')
    print(f'POCID: {pocid_result}')
    print(f'MASE: {mase_result}')

    return rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params
            
def run_times_fm(state, derivative, data_filtered, type_prediction='zeroshot', type_model='200M'):

    # Record the start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    try:
        rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
        create_times_fm(state=state,
                        derivative=derivative,
                        df=data_filtered, 
                        type_prediction=type_prediction, 
                        type_model=type_model)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'MODEL': 'TimesFM',
                                    'TYPE_MODEL': 'TimesFM_' + type_model,
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
        
        results_df = pd.DataFrame([{'MODEL': 'TimesFM',
                                    'TYPE_MODEL': 'TimesFM_' + type_model,
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

    file_path = os.path.join(directory, 'times_fm_results_last_year.xlsx')
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

def run_all_in_thread(type_prediction='zeroshot', type_model='200M'):

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
            
            run_times_fm(state, derivative, data_filtered, type_prediction, type_model)

def product_and_single_thread_testing():    
    state = "sp"
    derivative = "gasolinac"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/mensal/uf/{derivative}/mensal_{state}_{derivative}.csv", sep=";", parse_dates=['timestamp'], date_parser=convert_date)

    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()
    
    rrmse_result, mape_result, pbe_result, pocid_result, mase_result, y_pred, best_params = \
    create_times_fm(state=state,
                    derivative=derivative,
                    df=data_filtered_test,
                    type_prediction='fine_tuning_indiv',
                    type_model='200M') # 200M or 500M
    
    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")