import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from functools import partial
import optuna
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

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

def create_dataset_direct(data, time_steps=1, forecast_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps - forecast_steps + 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps:i + time_steps + forecast_steps, 0])
    return np.array(X), np.array(y)

def rolling_window(series, window, type_predictions):
    """
    Generate rolling window data for time series analysis.

    Parameters:
    - series: array-like, time series data
    - window: int, size of the rolling window

    Returns:
    - df: pandas DataFrame, containing the rolling window data
    - scaler: MinMaxScaler object, used for normalization
    """
    data = []

    if type_predictions == "recursive":
        for i in range(len(series) - window):
            example = np.array(series[i:i + window + 1])
            data.append(example)
    else:
        for i in range(len(series) - window):
            example = np.array(series[i:i + window + 12])
            data.append(example)

    df = pd.DataFrame(data)
    df = df.dropna()
    return df

def train_test_split_cisia(data, horizon, type_predictions):
    if type_predictions == "recursive":
        X = data.iloc[:,:-1] # features
        y = data.iloc[:,-1] # target

        X_train = X[:-horizon] # features train
        X_test =  X[-horizon:] # features test

        y_train = y[:-horizon] # target train
        y_test = y[-horizon:] # target test

    else: 
        X = data.iloc[:, :12]  
        y = data.iloc[:, 12:]  

        X_train = X[:-24]  # features train
        X_test = X[-1:]   # features test

        y_train = y[:-24]  # target train
        y_test = y[-1:]   # target test
        
    return X_train, X_test, y_train, y_test

def recursive_multistep_forecasting(X_test, model, horizon, device):
    model.eval() 
    example = torch.tensor(X_test[0].reshape(1, X_test.shape[1], X_test.shape[2]), dtype=torch.float32).to(device) 

    preds = []
    for i in range(horizon):
        with torch.no_grad():
            pred = model(example).cpu().numpy()[0]
        preds.append(pred)

        example = example[:, 1:, :] 
        pred_tensor = torch.tensor(pred.reshape(1, 1, -1), dtype=torch.float32).to(device)  
        example = torch.cat((example, pred_tensor), dim=1)  

    return np.array(preds)

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def objective(trial, device, X_train_val, X_val, y_train_val, y_test, df_mean, scaler, type_predictions):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.10])
    lr = trial.suggest_categorical("lr", [1e-3, 1e-4, 5e-5, 2e-5, 1e-6])

    if type_predictions == 'recursive':
        int_dense = 1
    else:
        int_dense = 12

    # DataLoader setup
    train_loader = DataLoader(TensorDataset(X_train_val, y_train_val), batch_size=batch_size, shuffle=True)

    model = LSTMModel(1, hidden_size, num_layers, int_dense, dropout).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/100 - Training Loss: {avg_loss:.6f}")

    # Forecasting and evaluation
    if type_predictions == "recursive":
        y_pred = scaler.inverse_transform(np.hstack([recursive_multistep_forecasting(X_val, model, 12, device).reshape(-1, 1), np.zeros((12, 12 - 1))]))[:, 0]
    else:
        model.eval()
        X_val = X_val[-1].to(device)

        with torch.no_grad():
            y_pred = model(X_val).cpu().numpy()  

        y_pred = scaler.inverse_transform(y_pred).flatten()
    
    rrmse_result_time_moe = rrmse(y_test, y_pred, df_mean)
    print(f'\n\nRRMSE: {rrmse_result_time_moe}')
    return rrmse_result_time_moe

def create_lstm_model(data, type_predictions='recursive'):
    """
    Creates and trains an LSTM model for time series forecasting using recursive or direct predictions.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing time series values.
    - type_predictions (str, optional): Specifies the prediction mode, either 'recursive' or 'direct'. Default is 'recursive'.

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

    # Data preparation based on prediction type
    if type_predictions == 'recursive':
        int_dense = 1

        # Prepare data for recursive predictions
        data = rolling_window(df, 12, type_predictions)
        X_train, X_test, y_train, y_test = train_test_split_cisia(data, 12, type_predictions)

        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        y_train = np.hstack([y_train, np.zeros((y_train.shape[0], 12 - 1))])
        y_test = np.hstack([y_test, np.zeros((y_test.shape[0], 12 - 1))])

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        y_train = scaler.transform(y_train)[:, 0].reshape(-1, 1)
        X_test = scaler.transform(X_test)
        y_test = scaler.transform(y_test)[:, 0].reshape(-1, 1)

    elif type_predictions == 'direct':
        int_dense = 12

        # Prepare data for direct predictions
        data = df.values.reshape(-1, 1)
        X, y = create_dataset_direct(data, 12, 12)

        X_train, X_test = X[:-24], X[-1:]
        y_train, y_test = y[:-24], y[-1:]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        y_train = scaler.transform(y_train)
        X_test = scaler.transform(X_test)
        y_test = scaler.transform(y_test)

    X_train_val, X_val = X_train[:-12], X_train[-12:]
    y_train_val, y_val = y_train[:-12], y_train[-12:]

    # Convert data to PyTorch tensors
    X_train_val = torch.tensor(X_train_val, dtype=torch.float32).unsqueeze(-1)
    y_train_val = torch.tensor(y_train_val, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # Set device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_seed(42)
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    objective_func = partial(objective, 
                             device = device,
                             X_train_val = X_train_val, 
                             X_val = X_val, 
                             y_train_val = y_train_val, 
                             y_test=df[:-12][-12:].values, 
                             df_mean=df[:-24].mean(),
                             scaler=scaler,
                             type_predictions= type_predictions)
    
    study.optimize(objective_func, n_trials=150)
    best_params = study.best_params

    input_size = 1
    num_layers = best_params["num_layers"]
    hidden_size = best_params["hidden_size"]
    output_size = int_dense
    dropout = best_params["dropout"]
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params["lr"])

    # DataLoader setup
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params["batch_size"], shuffle=False)

    # Training loop
    for epoch in range(100):
        model.train()
        total_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/100 - Training Loss: {avg_loss:.6f}")

    # Forecasting and evaluation
    if type_predictions == "recursive":
        y_pred = scaler.inverse_transform(np.hstack([recursive_multistep_forecasting(X_test, model, 12, device).reshape(-1, 1), np.zeros((12, 12 - 1))]))[:, 0]
    else:
        model.eval()

        X_test = X_test.to(device)

        with torch.no_grad():
            y_pred = model(X_test).cpu().numpy()  

        y_pred = scaler.inverse_transform(y_pred).flatten()

    y_test = df[-12:].values
    y_baseline = df[-12 * 2:-12].values

    # Evaluation metrics
    y_baseline = df[-12*2:-12].values
    rrmse_result_lstm = rrmse(y_test, y_pred, df[:-12].mean())
    mape_result_lstm = mape(y_test, y_pred)
    pbe_result_lstm = pbe(y_test, y_pred)
    pocid_result_lstm = pocid(y_test, y_pred)
    mase_result_lstm = np.mean(np.abs(y_test - y_pred)) / np.mean(np.abs(y_test - y_baseline))

    print(f"\nMetrics: LSTM_{type_predictions} \n")
    print(f'RRMSE: {rrmse_result_lstm}')
    print(f'MAPE: {mape_result_lstm}')
    print(f'PBE: {pbe_result_lstm}')
    print(f'POCID: {pocid_result_lstm}')
    print(f'MASE: {mase_result_lstm}')
        
    return rrmse_result_lstm, mape_result_lstm, pbe_result_lstm, pocid_result_lstm, mase_result_lstm, y_pred, best_params
                    
def run_lstm(state, product, data_filtered, type_predictions='recursive'):
    """
    Execute LSTM model training and save the results to an Excel file.

    Parameters:
        - state (str): State for which the LSTM model is trained.
        - product (str): Product for which the LSTM model is trained.
        - data_filtered (pd.DataFrame): Filtered dataset containing data for the specific state and product.
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct_dense12'). Default is 'recursive'.

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
        create_lstm_model(data=data_filtered, type_predictions=type_predictions)
        
        # Create a DataFrame to store the results
        results_df = pd.DataFrame([{'MODEL': 'LSTM',
                                    'TYPE_MODEL': 'LSTM',
                                    'TYPE_PREDICTIONS': 'LSTM_' + type_predictions,
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
                                    'TYPE_PREDICTIONS': 'LSTM_' + type_predictions,
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

    file_path = os.path.join(directory, 'lstm_results_pytorch.xlsx')
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

def run_lstm_in_thread(type_predictions='recursive'):
    """
    Execute LSTM model training in separate processes for different state and product combinations.

    Parameters:
        - type_predictions (str, optional): Type of prediction method ('recursive' or 'direct'). Default is 'recursive'.

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
                    data_filtered,
                    type_predictions
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
    create_lstm_model(data=data_filtered_test,
                      type_predictions='recursive')

    # Recording end time and calculating execution duration
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nFunction execution time: {execution_time:.2f} seconds")
    print(f"Execution ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")