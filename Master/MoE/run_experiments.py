# ======================================
# PARALLEL CONFIGURATION
# ======================================
from joblib import Parallel, delayed

# ======================================
# IMPORTS
# ======================================
import os
import json
import time
import torch
import numpy as np
import pandas as pd

from transformers import AutoModelForCausalLM
from chronos import BaseChronosPipeline
import timesfm

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from setup.models.modeling_model import predict_from_model

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, LSTM as NF_LSTM

from analysis_trained_models import run_analysis

# ======================================
# ENVIRONMENT CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ======================================
# GENERAL SETTINGS
# ======================================
file_path = "../dataset_global/dataset_global.jsonl"

HORIZONS = [3, 6, 12, 24]
YEARS = [2024, 2023, 2022, 2021, 2020]

BASE_CONTEXT = 410

base_path = "../all_datasets_global_by_years"


# ======================================
# HELPER: LAG-BASED RECURSIVE FORECAST
# ======================================
def _lag_recursive_forecast(train_np: np.ndarray, pred_len: int, n_lags: int, model_cls, **kwargs) -> np.ndarray:
    """
    Fits one model per series using lag features and forecasts recursively.

    Args:
        train_np : (n_series, context_length) float32 numpy array (already scaled).
        pred_len : number of steps to forecast.
        n_lags   : number of lag features.
        model_cls: sklearn-compatible regressor class.
        **kwargs : constructor arguments forwarded to model_cls.

    Returns:
        (n_series, pred_len) float32 numpy array.
    """
    n_series = train_np.shape[0]
    all_preds = np.zeros((n_series, pred_len), dtype=np.float32)

    for i in range(n_series):
        series = train_np[i].astype(np.float64)

        # Build supervised dataset from lag features
        X, y = [], []
        for j in range(n_lags, len(series)):
            X.append(series[j - n_lags:j])
            y.append(series[j])

        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        model = model_cls(**kwargs)
        model.fit(X, y)

        # Recursive multi-step prediction
        history = list(series[-n_lags:])
        preds = []
        for _ in range(pred_len):
            x_input = np.array(history[-n_lags:], dtype=np.float64).reshape(1, -1)
            p = model.predict(x_input)[0]
            preds.append(p)
            history.append(p)

        all_preds[i] = preds

    return all_preds


# ======================================
# HELPER: STATSFORECAST BATCH FORECAST
# ======================================
def _statsforecast_forecast(train_np: np.ndarray, pred_len: int, sf_models: list) -> dict:
    """
    Runs StatsForecast on all series and returns a dict {model_name: (n_series, pred_len) array}.

    Args:
        train_np  : (n_series, context_length) float32 numpy array (already scaled).
        pred_len  : forecast horizon.
        sf_models : list of statsforecast model instances.

    Returns:
        dict mapping each model's column name to a (n_series, pred_len) float32 array.
    """
    n_series, context_len = train_np.shape

    rows = []
    for i in range(n_series):
        for t in range(context_len):
            rows.append({"unique_id": str(i), "ds": t, "y": float(train_np[i, t])})
    df_sf = pd.DataFrame(rows)

    sf = StatsForecast(models=sf_models, freq=1, n_jobs=-1)
    forecast_df = sf.forecast(df=df_sf, h=pred_len)
    forecast_df = forecast_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    results = {}
    model_names = [m.__class__.__name__ for m in sf_models]
    for name in model_names:
        if name not in forecast_df.columns:
            # Some models add suffixes; find the matching column
            col = [c for c in forecast_df.columns if name in c]
            col = col[0] if col else None
        else:
            col = name
        if col is None:
            continue
        matrix = (
            forecast_df.groupby("unique_id")[col]
            .apply(list)
            .reset_index(drop=True)
        )
        results[name] = np.array(matrix.tolist(), dtype=np.float32)

    return results


# ======================================
# HELPER: NEURALFORECAST BATCH FORECAST
# ======================================
def _neuralforecast_forecast(
    train_np: np.ndarray,
    pred_len: int,
    input_size: int,
    max_steps: int = 100,
) -> dict:
    """
    Trains NBEATS and LSTM via NeuralForecast and returns predictions.

    Args:
        train_np   : (n_series, context_length) float32 numpy array (already scaled).
        pred_len   : forecast horizon.
        input_size : lookback window used by neural models.
        max_steps  : training steps (keep low for speed).

    Returns:
        dict {"NBEATS": array, "LSTM": array}, each (n_series, pred_len) float32.
    """
    n_series, context_len = train_np.shape

    rows = []
    for i in range(n_series):
        for t in range(context_len):
            rows.append({"unique_id": str(i), "ds": t, "y": float(train_np[i, t])})
    df_nf = pd.DataFrame(rows)

    models = [
        NBEATS(
            h=pred_len,
            input_size=input_size,
            max_steps=max_steps,
            accelerator="gpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
        NF_LSTM(
            h=pred_len,
            input_size=input_size,
            max_steps=max_steps,
            accelerator="gpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        ),
    ]

    nf = NeuralForecast(models=models, freq=1)
    nf.fit(df=df_nf)
    forecast_df = nf.predict()
    forecast_df = forecast_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    results = {}
    for col_name, key in [("NBEATS", "NBEATS"), ("LSTM", "LSTM")]:
        # NeuralForecast may append horizon info to column names
        matching = [c for c in forecast_df.columns if col_name in c]
        col = matching[0] if matching else None
        if col is None:
            continue
        matrix = (
            forecast_df.groupby("unique_id")[col]
            .apply(list)
            .reset_index(drop=True)
        )
        results[key] = np.array(matrix.tolist(), dtype=np.float32)

    return results


# ======================================
# MAIN PIPELINE
# ======================================
def run_full_experiment_pipeline(
    experiment_name: str,
    path_trained_models: str = "trained_models",
    top_k: int = 2,
    use_noise: bool = True,
):

    device = "cuda"

    N_CORES = 5
    debug_state = "sp"  # None

    results_root = f"results_by_state_year/{experiment_name}"
    times_root = f"times_by_state_year/{experiment_name}"

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(times_root, exist_ok=True)

    analysis_trained_log = True

    # ======================================
    # CONTEXT LENGTH FUNCTION
    # ======================================
    def get_context_length(year, horizon):
        return BASE_CONTEXT - horizon - ((2024 - year) * horizon)

    # ======================================
    # DATASET PROCESSING FUNCTION
    # ======================================
    def process_dataset(state_code, year, context_length, prediction_length):

        train_list, test_list, product_names = [], [], []

        with open(file_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                for key, value in entry.items():
                    if key.endswith(f"_{state_code}"):
                        total_len = context_length + prediction_length
                        if len(value) >= total_len:
                            seq = value[:total_len]
                            product_name = key.rsplit("_", 1)[0]
                            train_list.append(seq[:-prediction_length])
                            test_list.append(seq[-prediction_length:])
                            product_names.append(product_name)

        if len(train_list) == 0:
            print(f"No valid sequences for state={state_code}, year={year}")
            return None, None

        tensor_train = torch.tensor(train_list, dtype=torch.float32, device=device)
        tensor_test = torch.tensor(test_list, dtype=torch.float32, device=device)

        mean_vals = tensor_train.mean(dim=1, keepdim=True)
        std_vals = tensor_train.std(dim=1, keepdim=True)
        std_vals[std_vals == 0] = 1e-8

        tensor_train_scaled = (tensor_train - mean_vals) / std_vals
        tensor_train_scaled = tensor_train_scaled.to(device)

        # Numpy version for classical/ML models (CPU, already scaled)
        train_np = tensor_train_scaled.cpu().numpy()  # (n_series, context_length)
        std_np = std_vals.cpu().numpy()               # (n_series, 1)
        mean_np = mean_vals.cpu().numpy()             # (n_series, 1)

        times_dict = {}

        # ----------------------------------
        # Time-MoE
        # ----------------------------------
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M", trust_remote_code=True, device_map=device
        )
        out = model.generate(tensor_train_scaled, max_new_tokens=prediction_length)
        out = out[:, -prediction_length:]
        output_time_moe_200 = out * std_vals + mean_vals
        times_dict["Time-MoE200M"] = round(time.time() - start, 4)

        # ----------------------------------
        # Timer (Sundial)
        # ----------------------------------
        tensor_train_scaled_1d = tensor_train_scaled.squeeze(-1).to(device)

        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            device_map=device,
        )
        out = model.generate(tensor_train_scaled_1d, max_new_tokens=prediction_length)
        out = torch.as_tensor(out.squeeze(1))
        output_timer = out * std_vals + mean_vals
        times_dict["Timer"] = round(time.time() - start, 4)

        # ----------------------------------
        # TimesFM
        # ----------------------------------
        start = time.time()
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=prediction_length,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        with torch.no_grad():
            out, _ = model.forecast(tensor_train_scaled.cpu().numpy())
        output_timesfm = (
            torch.from_numpy(out).float().to(device) * std_vals + mean_vals
        )
        times_dict["TimesFM"] = round(time.time() - start, 4)

        # ----------------------------------
        # Moirai Small
        # ----------------------------------
        start = time.time()
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model.to(device)
        outs = []
        for i in range(tensor_train_scaled.size(0)):
            past = tensor_train_scaled[i].unsqueeze(0).unsqueeze(-1)
            obs = torch.ones_like(past, dtype=torch.bool)
            pad = torch.zeros_like(past, dtype=torch.bool).squeeze(-1)
            fc = model(past_target=past, past_observed_target=obs, past_is_pad=pad)
            outs.append(torch.as_tensor(fc.mean(dim=1)).reshape(1, -1))
        output_moirai_small = torch.cat(outs) * std_vals + mean_vals
        times_dict["Moirai-Small"] = round(time.time() - start, 4)

        # ----------------------------------
        # Chronos-Bolt-Small
        # ----------------------------------
        start = time.time()
        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        _, out = model.predict_quantiles(
            context=tensor_train_scaled, prediction_length=prediction_length
        )
        output_chronos_bolt_small = out.to(device) * std_vals + mean_vals
        times_dict["Chronos-Bolt-Small"] = round(time.time() - start, 4)

        # ----------------------------------
        # AutoETS (Nixtla / statsforecast)
        # ----------------------------------
        start = time.time()
        sf_preds = _statsforecast_forecast(
            train_np=train_np,
            pred_len=prediction_length,
            sf_models=[AutoETS(season_length=12)],
        )
        ets_np = sf_preds.get("AutoETS", np.zeros((train_np.shape[0], prediction_length), dtype=np.float32))
        output_autoets = (
            torch.from_numpy(ets_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["AutoETS"] = round(time.time() - start, 4)

        # ----------------------------------
        # AutoARIMA (Nixtla / statsforecast)
        # ----------------------------------
        start = time.time()
        sf_preds = _statsforecast_forecast(
            train_np=train_np,
            pred_len=prediction_length,
            sf_models=[AutoARIMA(season_length=12)],
        )
        arima_np = sf_preds.get("AutoARIMA", np.zeros((train_np.shape[0], prediction_length), dtype=np.float32))
        output_autoarima = (
            torch.from_numpy(arima_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["AutoARIMA"] = round(time.time() - start, 4)

        # ----------------------------------
        # N-BEATS + LSTM (Nixtla / neuralforecast)
        # ----------------------------------
        start = time.time()
        input_size = min(context_length, max(2 * prediction_length, 24))
        nf_preds = _neuralforecast_forecast(
            train_np=train_np,
            pred_len=prediction_length,
            input_size=input_size,
            max_steps=100,
        )

        nbeats_np = nf_preds.get("NBEATS", np.zeros((train_np.shape[0], prediction_length), dtype=np.float32))
        output_nbeats = (
            torch.from_numpy(nbeats_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["N-BEATS"] = round(time.time() - start, 4)

        start = time.time()
        lstm_np = nf_preds.get("LSTM", np.zeros((train_np.shape[0], prediction_length), dtype=np.float32))
        output_lstm = (
            torch.from_numpy(lstm_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["LSTM"] = round(time.time() - start, 4)

        # ----------------------------------
        # XGBRegressor (lag-based recursive)
        # ----------------------------------
        n_lags = min(context_length // 2, 48)

        start = time.time()
        xgb_np = _lag_recursive_forecast(
            train_np=train_np,
            pred_len=prediction_length,
            n_lags=n_lags,
            model_cls=XGBRegressor,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            verbosity=0,
            n_jobs=-1,
        )
        output_xgb = (
            torch.from_numpy(xgb_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["XGBRegressor"] = round(time.time() - start, 4)

        # ----------------------------------
        # RandomForest (lag-based recursive)
        # ----------------------------------
        start = time.time()
        rf_np = _lag_recursive_forecast(
            train_np=train_np,
            pred_len=prediction_length,
            n_lags=n_lags,
            model_cls=RandomForestRegressor,
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
        output_rf = (
            torch.from_numpy(rf_np).float().to(device) * std_vals + mean_vals
        )
        times_dict["RandomForest"] = round(time.time() - start, 4)

        # ----------------------------------
        # My-MoE
        # ----------------------------------
        start = time.time()
        model_path = (
            f"{path_trained_models}/horizon_{prediction_length}/"
            f"excluding_{state_code}/"
            f"{experiment_name}_{str(year)}.pt"
        )
        out = predict_from_model(
            model_path=model_path,
            series=tensor_train_scaled,
            horizon=prediction_length,
            context_length=context_length,
            top_k=top_k,
            use_noise="true" if use_noise else "false",
            device=device,
        )
        output_mymoe = out.to(device) * std_vals + mean_vals
        times_dict["My-MoE"] = round(time.time() - start, 4)

        # ----------------------------------
        # METRICS
        # ----------------------------------
        model_outputs = {
            "Time-MoE":          output_time_moe_200,
            "Timer":             output_timer,
            "TimesFM":           output_timesfm,
            "Moirai":            output_moirai_small,
            "Chronos":           output_chronos_bolt_small,
            "AutoETS":           output_autoets,
            "AutoARIMA":         output_autoarima,
            "N-BEATS":           output_nbeats,
            "LSTM":              output_lstm,
            "XGBRegressor":      output_xgb,
            "RandomForest":      output_rf,
            "My-MoE":            output_mymoe,
        }

        results = {}

        for name, preds in model_outputs.items():
            preds = torch.clamp(preds, min=0)
            mape_list = []

            for i in range(preds.shape[0]):
                mape = (
                    mean_absolute_percentage_error(
                        tensor_test[i].cpu().numpy(),
                        preds[i].cpu().numpy(),
                    )
                    * 100
                )
                mape_list.append(round(mape, 4))

            results[name] = mape_list

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.rename(columns={"index": "Modelo"}, inplace=True)
        df_results.columns = ["Modelo"] + [p.capitalize() for p in product_names]

        df_times = pd.DataFrame(
            list(times_dict.items()),
            columns=["Modelo", "Tempo (s)"],
        )

        return df_results, df_times

    # ======================================
    # PARALLEL WORKER
    # ======================================
    def run_single_experiment(
        state_code,
        year,
        horizon,
        context_length,
        results_path,
        times_path,
    ):
        print(
            f"Processing {state_code.upper()} - {year} "
            f"- Horizon {horizon} - Context {context_length}"
        )

        try:
            df_results, df_times = process_dataset(
                state_code=state_code,
                year=year,
                context_length=context_length,
                prediction_length=horizon,
            )

            if df_results is None or df_times is None:
                print(f"[SKIP] {state_code.upper()} - {year} - Horizon {horizon}")
                return None

            results_file = os.path.join(
                results_path, f"results_{state_code}_{year}.csv"
            )
            times_file = os.path.join(
                times_path, f"times_{state_code}_{year}.csv"
            )

            df_results.to_csv(results_file, index=False)
            df_times.to_csv(times_file, index=False)

            print(f"[DONE] {state_code.upper()} - {year} - Horizon {horizon}")
            return True

        except Exception as e:
            print(
                f"[ERROR] {state_code.upper()} - {year} "
                f"- Horizon {horizon}: {e}"
            )
            return None

    # ======================================
    # MAIN LOOP (PARALLEL)
    # ======================================
    for horizon in HORIZONS:
        print(f"\n===== Horizon {horizon} =====")

        results_path = os.path.join(results_root, f"horizon_{horizon}")
        times_path = os.path.join(times_root, f"horizon_{horizon}")

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(times_path, exist_ok=True)

        horizon_path = os.path.join(base_path, f"horizon_{horizon}")
        if not os.path.exists(horizon_path):
            continue

        tasks = []

        for excluding_folder in sorted(os.listdir(horizon_path)):
            state_code = excluding_folder.replace("excluding_", "")

            if debug_state is not None and state_code != debug_state:
                continue

            for year in YEARS:
                context_length = get_context_length(year, horizon)
                tasks.append(
                    (
                        state_code,
                        year,
                        horizon,
                        context_length,
                        results_path,
                        times_path,
                    )
                )

        Parallel(n_jobs=N_CORES, backend="loky", verbose=10)(
            delayed(run_single_experiment)(*task) for task in tasks
        )

    print("\nAll processing completed.")

    if analysis_trained_log:
        run_analysis(
            base_dir=path_trained_models,
            output_dir=f"output_dir/{os.path.basename(path_trained_models)}",
        )


# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    run_full_experiment_pipeline()