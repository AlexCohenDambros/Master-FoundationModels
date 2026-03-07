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
from setup.models.modeling_model import predict_from_model

from sklearn.metrics import mean_absolute_percentage_error
from analysis_trained_models import run_analysis
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA

from darts.models import NBEATSModel, RNNModel
from darts import TimeSeries

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

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
# MAIN PIPELINE
# ======================================
def run_full_experiment_pipeline(experiment_name: str, path_trained_models: str = "trained_models", top_k:int = 2, use_noise = True):

    device = "cuda"

    N_CORES = 5
    debug_state = None  # None or "sp"

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

        tensor_train = torch.tensor(
            train_list, dtype=torch.float32, device=device
        )
        tensor_test = torch.tensor(
            test_list, dtype=torch.float32, device=device
        )

        mean_vals = tensor_train.mean(dim=1, keepdim=True)
        std_vals = tensor_train.std(dim=1, keepdim=True)
        std_vals[std_vals == 0] = 1e-8

        tensor_train_scaled = (tensor_train - mean_vals) / std_vals
        tensor_train_scaled = tensor_train_scaled.to(device)

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
        # Timer
        # ----------------------------------
        tensor_train_scaled = tensor_train_scaled.squeeze(-1)
        tensor_train_scaled = tensor_train_scaled.to(device)

        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            device_map=device,
        )
        out = model.generate(
            tensor_train_scaled,
            max_new_tokens=prediction_length,
        )
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
            out, _ = model.forecast(
                tensor_train_scaled.cpu().numpy()
            )

        output_timesfm = (
            torch.from_numpy(out).float().to(device)
            * std_vals
            + mean_vals
        )

        times_dict["TimesFM"] = round(time.time() - start, 4)

        # -------- Moirai Small --------
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

        # -------- Chronos-Bolt-Small --------
        start = time.time()
        model = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device, torch_dtype=torch.bfloat16)
        _, out = model.predict_quantiles(
            context=tensor_train_scaled, prediction_length=prediction_length
        )
        output_chronos_bolt_small = out.to(device) * std_vals + mean_vals
        times_dict["Chronos-Bolt-Small"] = round(time.time() - start, 4)


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
                "y": series
            })

            sf = StatsForecast(
                models=[AutoETS(season_length=1)],
                freq=1,
                n_jobs=1
            )

            forecast = sf.forecast(df=df, h=prediction_length)
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
                "y": series
            })

            sf = StatsForecast(
                models=[AutoARIMA(season_length=1)],
                freq=1,
                n_jobs=1
            )

            forecast = sf.forecast(df=df, h=prediction_length)
            preds.append(forecast["AutoARIMA"].values)

        output_autoarima = torch.tensor(np.array(preds), device=device)
        times_dict["AutoARIMA"] = round(time.time() - start, 4)


        # -----------------------------
        # N-BEATS
        # -----------------------------
        start = time.time()
        preds = []

        input_len = context_length - prediction_length

        for i in range(n_series):

            series = TimeSeries.from_values(train_cpu[i])

            model = NBEATSModel(
            input_chunk_length=input_len,
            output_chunk_length=prediction_length,
            n_epochs=10,
            batch_size=32,
            random_state=42,
            )

            model.fit(series)

            forecast = model.predict(prediction_length)

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

            X = []
            y = []

            for t in range(context_length - prediction_length):
                X.append(series[t:t+prediction_length])
                y.append(series[t+prediction_length])

            X = np.array(X)
            y = np.array(y)

            model = RandomForestRegressor(n_estimators=200)

            model.fit(X, y)

            last_window = series[-prediction_length:]

            forecast = []

            window = last_window.copy()

            for _ in range(prediction_length):

                pred = model.predict(window.reshape(1,-1))[0]
                forecast.append(pred)

                window = np.roll(window,-1)
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

            X = []
            y = []

            for t in range(context_length - prediction_length):
                X.append(series[t:t+prediction_length])
                y.append(series[t+prediction_length])

            X = np.array(X)
            y = np.array(y)

            model = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05
            )

            model.fit(X, y)

            last_window = series[-prediction_length:]

            forecast = []
            window = last_window.copy()

            for _ in range(prediction_length):

                pred = model.predict(window.reshape(1,-1))[0]
                forecast.append(pred)

                window = np.roll(window,-1)
                window[-1] = pred

            preds.append(forecast)

        output_xgb = torch.tensor(np.array(preds), device=device)
        times_dict["XGBRegressor"] = round(time.time() - start, 4)

        # ----------------------------------
        # METRICS
        # ----------------------------------
        model_outputs = {
            "Time-MoE": output_time_moe_200,
            "Timer": output_timer,
            "TimesFM": output_timesfm,
            "Morai": output_moirai_small,
            "Chronos": output_chronos_bolt_small,
            "My-MoE": output_mymoe,

            "AutoETS": output_autoets,
            "AutoARIMA": output_autoarima,
            "NBEATS": output_nbeats,
            "RandomForest": output_rf,
            "XGBRegressor": output_xgb,
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
        df_results.columns = ["Modelo"] + [
            p.capitalize() for p in product_names
        ]

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
                print(
                    f"[SKIP] {state_code.upper()} - {year} "
                    f"- Horizon {horizon}"
                )
                return None

            results_file = os.path.join(
                results_path,
                f"results_{state_code}_{year}.csv",
            )
            times_file = os.path.join(
                times_path,
                f"times_{state_code}_{year}.csv",
            )

            df_results.to_csv(results_file, index=False)
            df_times.to_csv(times_file, index=False)

            print(
                f"[DONE] {state_code.upper()} - {year} "
                f"- Horizon {horizon}"
            )

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

        results_path = os.path.join(
            results_root, f"horizon_{horizon}"
        )
        times_path = os.path.join(
            times_root, f"horizon_{horizon}"
        )

        os.makedirs(results_path, exist_ok=True)
        os.makedirs(times_path, exist_ok=True)

        horizon_path = os.path.join(
            base_path, f"horizon_{horizon}"
        )
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

        Parallel(
            n_jobs=N_CORES,
            backend="loky",
            verbose=10,
        )(
            delayed(run_single_experiment)(*task)
            for task in tasks
        )


    print("\nAll processing completed.")

    if analysis_trained_log:
        run_analysis(base_dir=path_trained_models, output_dir=f"output_dir/{os.path.basename(path_trained_models)}",)


# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    run_full_experiment_pipeline()