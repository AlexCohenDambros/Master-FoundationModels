from itertools import product
import torch
from transformers import AutoModelForCausalLM
from time_moe.models.modeling_time_moe import TimeMoeForPrediction

import tensorflow as tf
import glob

from sklearn.metrics import mean_absolute_percentage_error as mape

import numpy as np
import pandas as pd

import os
import random
import time

import gc
import multiprocessing

import warnings
from warnings import simplefilter

from metrics_time_moe import pbe # type: ignore
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

def get_scaled_data(df):
    df = df[:-1]

    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values.reshape(-1, 1))

    return df_scaled, scaler


def create_timemoe_model(
    data, 
    state, 
    derivative, 
    type_model,
):

    y_preds_5_years = []

    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

    end_date = data['timestamp'].max()

    for years in range(5, 0, -1):
        start_date = end_date - pd.DateOffset(years=years-1)

        df = data[data['timestamp'] <= start_date]
        print(f'\nData filtered for {start_date.date()}\n')
        print(df)
        df = df['m3']

        print(f"Ano atual: {(start_date.date()).year}")
   
        df_scaled, scaler = get_scaled_data(df)

        tensor = torch.tensor(df_scaled, dtype=torch.float32)

        tensor = tensor.squeeze(-1).unsqueeze(0)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tensor = tensor.to(device)

        ''' 
        # INFO: ================== ZERO SHOT ==================
        ''' 
        if type_model == "TimeMoE-50M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-50M',
                device_map="cuda",  
                trust_remote_code=True,
            )

        elif type_model == "TimeMoE-200M_ZERO_SHOT":
            model = AutoModelForCausalLM.from_pretrained(
                'Maple728/TimeMoE-200M',
                device_map="cuda",  
                trust_remote_code=True,
            )

        model.to(device)

        # forecast
        prediction_length = 1
        output = model.generate(tensor, max_new_tokens=prediction_length)  
        forecast = output[:, -prediction_length:]  

        forecast = forecast.cpu().numpy() 
    
        gc.collect() 

        y_pred = scaler.inverse_transform(forecast)
        y_pred = y_pred.flatten()

        y_test = df[-1:].values

        # Calculating evaluation metrics
        mape_result_timemoe = mape(y_test, y_pred)
        pbe_result_timemoe = pbe(y_test, y_pred)

        print("\nResultados Time-MOE: \n")
        print(f'MAPE: {mape_result_timemoe}')
        print(f'PBE: {pbe_result_timemoe}')
      
        y_preds_5_years.append(y_pred)

    return np.concatenate(y_preds_5_years).tolist()
                    
def run_timemoe_5_years(state, derivative, city, data_filtered, type_model):

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
        y_pred = create_timemoe_model(
            data=data_filtered,
            state=state,
            derivative=derivative,
            type_model=type_model
        )
        # Add results to DataFrame
        results_df = pd.DataFrame([{
            'TYPE_PREDICTIONS': type_model,
            'STATE': state,
            'CITY': city,
            'PRODUCT': derivative,
            'PREDICTIONS': y_pred,
            'PARAMETERS': None,
            'ERROR': np.nan
        }])

    except Exception as e:
        # Handle any exceptions during model training
        print(f"An error occurred for derivative '{derivative}' in state '{state}': {e}")

        results_df = pd.DataFrame([{
            'TYPE_PREDICTIONS': type_model,
            'STATE': state,
            'CITY': city,
            'PRODUCT': derivative,
            'PREDICTIONS': np.nan,
            'PARAMETERS': None,
            'ERROR': f"An error occurred for derivative '{derivative}' in state '{state}': {e}"
        }])

    # Save results to an Excel file if specified
    directory = f'results_model_local'
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, 'results_timemoe_cities_5_years.xlsx')
    if os.path.exists(file_path):
        existing_df = pd.read_excel(file_path)
    else:
        existing_df = pd.DataFrame()

    combined_df = pd.concat([existing_df, results_df], ignore_index=True)
    combined_df.to_excel(file_path, index=False)

    # Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Function execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def run_all_in_thread_5_years_for_cities(type_model):
    multiprocessing.set_start_method("spawn")

    directory_path = "../database/venda_process/anual/municipio/*/"

    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    state_product_city_dict = {}

    for file in csv_files:
        parts = os.path.basename(file).split("_")
        if len(parts) < 4:
            continue  
        
        _, city, state, derivative = parts[:4]  
        derivative = derivative.replace(".csv", "")  
        
        data = pd.read_csv(file, sep=";", parse_dates=['timestamp'])
        
        if state not in state_product_city_dict:
            state_product_city_dict[state] = {}
        if derivative not in state_product_city_dict[state]:
            state_product_city_dict[state][derivative] = []
        state_product_city_dict[state][derivative].append((city, data))

    for state, products in state_product_city_dict.items():
        for derivative, city_data_list in products.items():
            for city, data_filtered in city_data_list:
                if len(data_filtered) > 12:
                    print(data_filtered)
                    print(f"========== State: {state}, Product: {derivative}, City: {city} ==========")

                    random.seed(42)
                    np.random.seed(42)
                    tf.random.set_seed(42)
                    os.environ['PYTHONHASHSEED'] = str(42)
                    tf.keras.utils.set_random_seed(42)

                    thread = multiprocessing.Process(target=run_timemoe_5_years, args=(state, derivative, city, data_filtered, type_model))
                    thread.start()
                    thread.join()

def product_and_single_thread_testing_5_years_for_cities():    
    """
    Perform a simple training thread using timemoe model for time series forecasting.

    This function initializes random seeds, loads a database, executes an timemoe model,
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

    state = "pr"
    derivative = "etanol"
    city = "pontagrossa"
    
    # Loading and preparing data
    data_filtered_test = pd.read_csv(f"../database/venda_process/anual/municipio/{derivative}/anual_{city}_{state}_{derivative}.csv", sep=";",  parse_dates=['timestamp'])
    
    print(f" ========== Starting univariate test for the state of {state} - {derivative} ==========")

    # Recording start time of execution
    start_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nExecution started at: {start_timestamp}")
    start_time = time.time()

    # Running the timemoe model
    y_pred = create_timemoe_model(
                data= data_filtered_test,  
                state= state, 
                derivative= derivative,
                type_model='TimeMoE-200M_ZERO_SHOT'
     )

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")