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
import pandas as pd

from transformers import AutoModelForCausalLM
from chronos import BaseChronosPipeline
import timesfm

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from setup.models.modeling_model_encoder import predict_from_model as mymoe_predict

from sklearn.metrics import mean_absolute_percentage_error

import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

from darts.models import NBEATSModel
from darts import TimeSeries

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST as NF_PatchTST, iTransformer as NF_iTransformer

# ======================================
# ENVIRONMENT CONFIGURATION
# ======================================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["WANDB_MODE"] = "disabled"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ======================================
# GENERAL SETTINGS
# ======================================
TEST_BASE_PATH = "../benchmark_prepared_test"


# ======================================
# DATA LOADING
# ======================================
def _load_test_data(dataset_name: str, horizon: int, subfolder: str):
    """Load all series from benchmark_prepared_test and split into train/test tensors.

    Series with different lengths are truncated to the shortest one so that
    they can be stacked into a single batch tensor.
    """
    file_path = os.path.join(TEST_BASE_PATH, dataset_name, subfolder, "dataset.jsonl")

    if not os.path.isfile(file_path):
        print(f"  [WARNING] Test data not found: {file_path}", flush=True)
        return None, None

    train_list, test_list = [], []

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            for _, value in entry.items():
                if len(value) >= horizon + 1:
                    train_list.append(value[:-horizon])
                    test_list.append(value[-horizon:])

    if not train_list:
        print(f"  [WARNING] No valid sequences for {dataset_name}", flush=True)
        return None, None

    # Truncate all train sequences to the minimum length so they batch cleanly
    min_len = min(len(s) for s in train_list)
    train_list = [s[-min_len:] for s in train_list]

    tensor_train = torch.tensor(train_list, dtype=torch.float32)
    tensor_test  = torch.tensor(test_list,  dtype=torch.float32)
    return tensor_train, tensor_test


# ======================================
# MAIN PIPELINE
# ======================================
def run_benchmark_experiment_pipeline(
    experiment_name: str,
    trained_models_root: str,
    datasets: dict,
    top_k: int = 2,
    use_noise: bool = True,
    device: str = "cuda",
    n_jobs: int = 1,
):
    """Run the full prediction pipeline on benchmark_prepared_test for every dataset.

    Results are saved to:
        results_by_state_year/{dataset_name}/{experiment_name}/results_{dataset_name}.csv
        times_by_state_year/{dataset_name}/{experiment_name}/times_{dataset_name}.csv
    """

    # ======================================
    # DATASET PROCESSING FUNCTION
    # ======================================
    def process_dataset(dataset_name: str, horizon: int, subfolder: str):

        tensor_train, tensor_test = _load_test_data(dataset_name, horizon, subfolder)
        if tensor_train is None or tensor_test is None:
            return None, None

        context_length = tensor_train.shape[1]

        mean_vals = tensor_train.mean(dim=1, keepdim=True)
        std_vals  = tensor_train.std(dim=1, keepdim=True)
        std_vals[std_vals == 0] = 1e-8

        tensor_train_scaled = ((tensor_train - mean_vals) / std_vals).to(device)
        mean_vals  = mean_vals.to(device)
        std_vals   = std_vals.to(device)
        tensor_test = tensor_test.to(device)

        times_dict = {}

        # ----------------------------------
        # Time-MoE
        # ----------------------------------
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M", trust_remote_code=True, device_map=device
        )
        out = model.generate(tensor_train_scaled, max_new_tokens=horizon)
        out = out[:, -horizon:]
        output_time_moe_200 = out * std_vals + mean_vals
        times_dict["Time-MoE200M"] = round(time.time() - start, 4)

        # ----------------------------------
        # Timer (Sundial)
        # ----------------------------------
        tensor_train_scaled_sq = tensor_train_scaled.squeeze(-1).to(device)

        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            device_map=device,
        )
        out = model.generate(tensor_train_scaled_sq, max_new_tokens=horizon)
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
                horizon_len=horizon,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )
        with torch.no_grad():
            out, _ = model.forecast(tensor_train_scaled.cpu().numpy().tolist())
        output_timesfm = (
            torch.from_numpy(out).float().to(device) * std_vals + mean_vals
        )
        times_dict["TimesFM"] = round(time.time() - start, 4)

        # -------- Moirai Small --------
        start = time.time()
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=horizon,
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
            obs  = torch.ones_like(past, dtype=torch.bool)
            pad  = torch.zeros_like(past, dtype=torch.bool).squeeze(-1)
            fc   = model(past_target=past, past_observed_target=obs, past_is_pad=pad)
            outs.append(torch.as_tensor(fc.mean(dim=1)).reshape(1, -1))
        output_moirai_small = torch.cat(outs) * std_vals + mean_vals
        times_dict["Moirai-Small"] = round(time.time() - start, 4)

        # -------- Chronos-Bolt-Small --------
        start = time.time()
        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small", device_map=device, torch_dtype=torch.bfloat16
        )
        _, out = model.predict_quantiles(
            context=tensor_train_scaled, prediction_length=horizon
        )
        output_chronos_bolt_small = out.to(device) * std_vals + mean_vals
        times_dict["Chronos-Bolt-Small"] = round(time.time() - start, 4)

        # ----------------------------------
        # FM-MoE
        # ----------------------------------
        start = time.time()
        model_path = os.path.join(
            trained_models_root, dataset_name, "best_model",
            experiment_name, f"horizon_{horizon}",
            f"model_{experiment_name}.pt",
        )
        try:
            out = mymoe_predict(
                model_path=model_path,
                series=tensor_train_scaled,
                horizon=horizon,
                top_k=top_k,
                use_noise="true" if use_noise else "false",
                device=device,
            )
            output_mymoe = out.to(device) * std_vals + mean_vals
        except Exception as e:
            print(f"  [ERROR] FM-MoE prediction failed for {dataset_name}: {e}", flush=True)
            output_mymoe = torch.zeros_like(tensor_test)
        times_dict["FM-MoE"] = round(time.time() - start, 4)

        # ==================================
        # CLASSICAL / ML MODELS
        # ==================================

        train_cpu = tensor_train.cpu().numpy()
        n_series = train_cpu.shape[0]

        # -----------------------------
        # AutoETS
        # -----------------------------
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            df = pd.DataFrame({
                "unique_id": "series",
                "ds": np.arange(len(series)),
                "y": series,
            })
            sf = StatsForecast(models=[AutoETS(season_length=1)], freq=1, n_jobs=1)
            forecast = sf.forecast(df=df, h=horizon)
            preds.append(forecast["AutoETS"].values)
        output_autoets = torch.tensor(np.array(preds), device=device)
        times_dict["AutoETS"] = round(time.time() - start, 4)

        # -----------------------------
        # AutoARIMA
        # -----------------------------
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            df = pd.DataFrame({
                "unique_id": "series",
                "ds": np.arange(len(series)),
                "y": series,
            })
            sf = StatsForecast(models=[AutoARIMA(season_length=1)], freq=1, n_jobs=1)
            forecast = sf.forecast(df=df, h=horizon)
            preds.append(forecast["AutoARIMA"].values)
        output_autoarima = torch.tensor(np.array(preds), device=device)
        times_dict["AutoARIMA"] = round(time.time() - start, 4)

        # -----------------------------
        # N-BEATS
        # -----------------------------
        start = time.time()
        preds = []
        input_len = context_length - horizon
        for i in range(n_series):
            series = TimeSeries.from_values(train_cpu[i])
            model = NBEATSModel(
                input_chunk_length=input_len,
                output_chunk_length=horizon,
                n_epochs=10,
                batch_size=32,
                random_state=42,
            )
            model.fit(series)
            forecast = model.predict(horizon)
            preds.append(forecast.values().flatten())
        output_nbeats = torch.tensor(np.array(preds), device=device)
        times_dict["NBEATS"] = round(time.time() - start, 4)

        # -----------------------------
        # Random Forest
        # -----------------------------
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            X, y = [], []
            for t in range(context_length - horizon):
                X.append(series[t:t+horizon])
                y.append(series[t+horizon])
            X, y = np.array(X), np.array(y)
            model = RandomForestRegressor(n_estimators=200)
            model.fit(X, y)
            window = series[-horizon:].copy()
            forecast = []
            for _ in range(horizon):
                pred = model.predict(window.reshape(1, -1))[0]
                forecast.append(pred)
                window = np.roll(window, -1)
                window[-1] = pred
            preds.append(forecast)
        output_rf = torch.tensor(np.array(preds), device=device)
        times_dict["RandomForest"] = round(time.time() - start, 4)

        # -----------------------------
        # XGBoost
        # -----------------------------
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            X, y = [], []
            for t in range(context_length - horizon):
                X.append(series[t:t+horizon])
                y.append(series[t+horizon])
            X, y = np.array(X), np.array(y)
            model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05)
            model.fit(X, y)
            window = series[-horizon:].copy()
            forecast = []
            for _ in range(horizon):
                pred = model.predict(window.reshape(1, -1))[0]
                forecast.append(pred)
                window = np.roll(window, -1)
                window[-1] = pred
            preds.append(forecast)
        output_xgb = torch.tensor(np.array(preds), device=device)
        times_dict["XGBRegressor"] = round(time.time() - start, 4)


        # -----------------------------
        # PatchTST
        # -----------------------------
        start = time.time()
        panel_df = pd.DataFrame({
            "unique_id": np.repeat(np.arange(n_series), context_length),
            "ds": np.tile(np.arange(context_length), n_series),
            "y": train_cpu.flatten(),
        })
        nf_pt = NeuralForecast(
            models=[NF_PatchTST(input_size=context_length, h=horizon, max_steps=100, scaler_type="standard")],
            freq=1,
        )
        nf_pt.fit(panel_df)
        fc_pt = nf_pt.predict()
        if not isinstance(fc_pt, pd.DataFrame):
            fc_pt = fc_pt.to_pandas()
        fc_pt = fc_pt.sort_values(["unique_id", "ds"])
        output_patchtst = torch.tensor(
            fc_pt["PatchTST"].to_numpy().reshape(n_series, horizon),
            dtype=torch.float32, device=device,
        )
        times_dict["PatchTST"] = round(time.time() - start, 4)


        # -----------------------------
        # iTransformer
        # -----------------------------
        start = time.time()
        nf_it = NeuralForecast(
            models=[NF_iTransformer(input_size=context_length, h=horizon, n_series=n_series, max_steps=100, scaler_type="standard")],
            freq=1,
        )
        nf_it.fit(panel_df)
        fc_it = nf_it.predict()
        if not isinstance(fc_it, pd.DataFrame):
            fc_it = fc_it.to_pandas()
        fc_it = fc_it.sort_values(["unique_id", "ds"])
        output_itransformer = torch.tensor(
            fc_it["iTransformer"].to_numpy().reshape(n_series, horizon),
            dtype=torch.float32, device=device,
        )
        times_dict["iTransformer"] = round(time.time() - start, 4)

        # ----------------------------------
        # METRICS
        # ----------------------------------
        model_outputs = {
            "Time-MoE": output_time_moe_200,
            "Timer":    output_timer,
            "TimesFM":  output_timesfm,
            "Moirai":   output_moirai_small,
            "Chronos":  output_chronos_bolt_small,
            "FM-MoE":   output_mymoe,
            "AutoETS":      output_autoets,
            "AutoARIMA":    output_autoarima,
            "NBEATS":       output_nbeats,
            "RandomForest": output_rf,
            "XGBRegressor": output_xgb,
            "PatchTST":     output_patchtst,
            "iTransformer": output_itransformer,
        }

        ensemble_models = {k: v for k, v in model_outputs.items() if k != "FM-MoE"}
        stacked_outputs  = torch.stack(list(ensemble_models.values()), dim=0)
        ensemble_output  = torch.mean(stacked_outputs, dim=0)
        model_outputs["Ensemble"] = ensemble_output

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
                mape_list.append(round(float(mape), 4))
            results[name] = mape_list

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.rename(columns={"index": "Modelo"}, inplace=True)

        df_times = pd.DataFrame(
            list(times_dict.items()),
            columns=["Modelo", "Tempo (s)"],
        )

        return df_results, df_times

    # ======================================
    # WORKER (used for optional parallelism)
    # ======================================
    def run_single_dataset(dataset_name: str, horizon: int, subfolder: str):
        print(
            f"Processing {dataset_name} | horizon={horizon}",
            flush=True,
        )

        results_path = os.path.join(
            "results_by_state_year", dataset_name, experiment_name
        )
        times_path = os.path.join(
            "times_by_state_year", dataset_name, experiment_name
        )
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(times_path, exist_ok=True)

        try:
            df_results, df_times = process_dataset(dataset_name, horizon, subfolder)

            if df_results is None or df_times is None:
                print(f"[SKIP] {dataset_name} — no data found", flush=True)
                return None

            results_file = os.path.join(results_path, f"results_{dataset_name}.csv")
            times_file   = os.path.join(times_path,   f"times_{dataset_name}.csv")

            df_results.to_csv(results_file, index=False)
            df_times.to_csv(times_file,     index=False)

            print(f"[DONE] {dataset_name} → {results_file}", flush=True)
            return True

        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}", flush=True)
            return None

    # ======================================
    # MAIN LOOP
    # ======================================
    tasks = [
        (name, cfg["horizon"], cfg["subfolder"])
        for name, cfg in datasets.items()
    ]

    if n_jobs == 1:
        for name, horizon, subfolder in tasks:
            run_single_dataset(name, horizon, subfolder)
    else:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(run_single_dataset)(name, horizon, subfolder)
            for name, horizon, subfolder in tasks
        )

    print("\nAll benchmark experiments completed.", flush=True)