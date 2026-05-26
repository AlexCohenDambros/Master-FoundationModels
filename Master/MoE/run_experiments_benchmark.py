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
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoARIMA, SeasonalNaive

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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# ======================================
# BENCHMARK DATASET REGISTRY
# ======================================
BASE_TEST_PATH = "../benchmark_prepared_test"

DATASETS = {
    # "cif_2016_filtered":        {"subfolder": "horizon_12",  "horizon": 12, "context_length": 96},
    # "etth_filtered":            {"subfolder": "horizon_36",  "horizon": 36, "context_length": 102},
    # "hospital_filtered":        {"subfolder": "horizon_12",  "horizon": 12, "context_length": 60},
    # "m3_monthly_filtered":      {"subfolder": "horizon_18",  "horizon": 18, "context_length": 98},
    # "m4_monthly_filtered":      {"subfolder": "horizon_18",  "horizon": 18, "context_length": 51},
    # "nn5_weekly_filtered":      {"subfolder": "horizon_8",   "horizon": 8,  "context_length": 97},
    # "tourism_monthly_filtered": {"subfolder": "horizon_24",  "horizon": 24, "context_length": 285},
    # "weather_filtered":         {"subfolder": "horizon_36",  "horizon": 36, "context_length": 454},
    "fred_md_filtered":           {"subfolder": "horizon_12",  "horizon": 12, "context_length": 704},
    "m5_filtered":                {"subfolder": "horizon_28",  "horizon": 28, "context_length": 1913},
}

# ======================================
# DATA LOADING
# ======================================
def load_benchmark_dataset(dataset_name, cfg):
    horizon        = cfg["horizon"]
    context_length = cfg["context_length"]
    required_len   = context_length + horizon

    jsonl_path = os.path.join(BASE_TEST_PATH, dataset_name, cfg["subfolder"], "dataset.jsonl")

    if not os.path.isfile(jsonl_path):
        print(f"[WARN] Missing data file: {jsonl_path}", flush=True)
        return None, None

    input_list, target_list = [], []

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            series = json.loads(line)["sequence"]
            if len(series) < required_len:
                print(
                    f"[WARN] {dataset_name} line {line_num}: "
                    f"length {len(series)} < required {required_len}, skipping",
                    flush=True,
                )
                continue
            input_list.append(series[-(horizon + context_length):-horizon])
            target_list.append(series[-horizon:])

    if not input_list:
        print(f"[WARN] {dataset_name}: no valid series after filtering", flush=True)
        return None, None

    return input_list, target_list


# ======================================
# MAIN PIPELINE
# ======================================
def run_full_experiment_pipeline(
    path_trained_models: str,
    experiment_name: str,
    top_k: int = 2,
    use_noise: bool = True,
):
    device = "cuda"

    results_root = os.path.join("results_benchmark", os.path.basename(path_trained_models))
    times_root   = os.path.join("times_benchmark",   os.path.basename(path_trained_models))

    os.makedirs(results_root, exist_ok=True)
    os.makedirs(times_root,   exist_ok=True)

    # ======================================
    # DATASET PROCESSING
    # ======================================
    def process_dataset(dataset_name, cfg, input_list, target_list):
        horizon        = cfg["horizon"]
        context_length = cfg["context_length"]
        print("context_length:", context_length)
        print("horizon:", horizon)

        tensor_train = torch.tensor(input_list, dtype=torch.float32, device=device)
        tensor_test  = torch.tensor(target_list, dtype=torch.float32, device=device)

        mean_vals = tensor_train.mean(dim=1, keepdim=True)
        std_vals  = tensor_train.std(dim=1, keepdim=True)
        std_vals[std_vals == 0] = 1e-8

        tensor_train_scaled = (tensor_train - mean_vals) / std_vals
        tensor_train_scaled = tensor_train_scaled.to(device)

        times_dict = {}

        # ----------------------------------
        # Time-MoE
        # ----------------------------------
        print("Start Time-MoE...")
        start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M", trust_remote_code=True, device_map=device
        )
        out = model.generate(tensor_train_scaled, max_new_tokens=horizon)
        out = out[:, -horizon:]
        output_time_moe_200 = out * std_vals + mean_vals
        times_dict["Time-MoE200M"] = round(time.time() - start, 4)

        # ----------------------------------
        # Timer
        # ----------------------------------
        print("Start Timer...")
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
            max_new_tokens=horizon,
        )
        out = torch.as_tensor(out.squeeze(1))
        output_timer = out * std_vals + mean_vals
        times_dict["Timer"] = round(time.time() - start, 4)

        # ----------------------------------
        # TimesFM
        # ----------------------------------
        print("Start TimesFM...")
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
            out, _ = model.forecast(tensor_train_scaled.cpu().numpy())

        output_timesfm = (
            torch.from_numpy(out).float().to(device) * std_vals + mean_vals
        )
        times_dict["TimesFM"] = round(time.time() - start, 4)

        # ----------------------------------
        # Moirai-Small
        # ----------------------------------
        print("Start Moirai...")
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

        # ----------------------------------
        # Chronos-Bolt-Small
        # ----------------------------------
        print("Start Chronos...")
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
        print("Start FM-MoE...")
        start = time.time()
        model_path = os.path.join(path_trained_models, dataset_name, f"{experiment_name}.pt")

        if not os.path.isfile(model_path):
            print(f"[WARN] FM-MoE model not found: {model_path}. Filling with zeros.", flush=True)
            output_mymoe = torch.zeros((tensor_train_scaled.shape[0], horizon), device=device)
        else:
            try:
                out = predict_from_model(
                    model_path=model_path,
                    series=tensor_train_scaled,
                    horizon=horizon,
                    context_length=context_length,
                    top_k=top_k,
                    use_noise="true" if use_noise else "false",
                    device=device,
                )
                output_mymoe = out.to(device) * std_vals + mean_vals
            except Exception as e:
                print(f"[ERROR] FM-MoE prediction failed for {dataset_name}: {e}", flush=True)
                output_mymoe = torch.zeros((tensor_train_scaled.shape[0], horizon), device=device)

        times_dict["FM-MoE"] = round(time.time() - start, 4)

        # ==================================
        # CLASSICAL / ML MODELS
        # ==================================
        train_cpu = tensor_train.cpu().numpy()
        n_series  = train_cpu.shape[0]

        # ----------------------------------
        # AutoETS
        # ----------------------------------
        print("Start AutoETS...")
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

        # ----------------------------------
        # AutoARIMA
        # ----------------------------------
        print("Start AutoARIMA...")
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

        # ----------------------------------
        # SeasonalNaive
        # ----------------------------------
        print("Start SeasonalNaive...")
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            df = pd.DataFrame({
                "unique_id": "series",
                "ds": np.arange(len(series)),
                "y": series,
            })
            sf = StatsForecast(models=[SeasonalNaive(season_length=12)], freq=1, n_jobs=1)
            forecast = sf.forecast(df=df, h=horizon)
            preds.append(forecast["SeasonalNaive"].values)
        output_seasonal_naive = torch.tensor(np.array(preds), device=device)
        times_dict["SeasonalNaive"] = round(time.time() - start, 4)

        # ----------------------------------
        # N-BEATS
        # ----------------------------------
        print("Start N-BEATS...")
        start = time.time()
        preds = []
        for i in range(n_series):
            series = TimeSeries.from_values(train_cpu[i])
            model = NBEATSModel(
                input_chunk_length=context_length - horizon,
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

        # ----------------------------------
        # Random Forest
        # ----------------------------------
        print("Start RF...")
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            X, y = [], []
            for t in range(context_length - horizon):
                X.append(series[t:t + horizon])
                y.append(series[t + horizon])
            X = np.array(X)
            y = np.array(y)
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

        # ----------------------------------
        # XGBoost
        # ----------------------------------
        print("Start XGBoost...")
        start = time.time()
        preds = []
        for i in range(n_series):
            series = train_cpu[i]
            X, y = [], []
            for t in range(context_length - horizon):
                X.append(series[t:t + horizon])
                y.append(series[t + horizon])
            X = np.array(X)
            y = np.array(y)
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

        # ----------------------------------
        # PatchTST
        # ----------------------------------
        nf_input_size = max(8, min(context_length // 2, context_length - horizon - 1))

        print(f"Start PatchTST... (input_size={nf_input_size}, h={horizon})")
        start = time.time()
        panel_df = pd.DataFrame({
            "unique_id": np.repeat(np.arange(n_series), context_length),
            "ds":        np.tile(np.arange(context_length), n_series),
            "y":         train_cpu.flatten(),
        })
        nf_pt = NeuralForecast(
            models=[NF_PatchTST(input_size=nf_input_size, h=horizon, max_steps=100)],
            freq=1,
        )
        nf_pt.fit(panel_df, val_size=0)
        fc_pt = nf_pt.predict()
        if not isinstance(fc_pt, pd.DataFrame):
            fc_pt = fc_pt.to_pandas()
        fc_pt = fc_pt.sort_values(["unique_id", "ds"])
        output_patchtst = torch.tensor(
            fc_pt["PatchTST"].to_numpy().reshape(n_series, horizon),
            dtype=torch.float32, device=device,
        )
        times_dict["PatchTST"] = round(time.time() - start, 4)

        # ----------------------------------
        # iTransformer
        # ----------------------------------
        print(f"Start iTransformer... (input_size={nf_input_size}, h={horizon})")
        start = time.time()
        nf_it = NeuralForecast(
            models=[NF_iTransformer(input_size=nf_input_size, h=horizon, n_series=n_series, max_steps=100)],
            freq=1,
        )
        nf_it.fit(panel_df, val_size=0)
        fc_it = nf_it.predict()
        if not isinstance(fc_it, pd.DataFrame):
            fc_it = fc_it.to_pandas()
        fc_it = fc_it.sort_values(["unique_id", "ds"])
        output_itransformer = torch.tensor(
            fc_it["iTransformer"].to_numpy().reshape(n_series, horizon),
            dtype=torch.float32, device=device,
        )
        times_dict["iTransformer"] = round(time.time() - start, 4)

        # ==================================
        # METRICS
        # ==================================
        model_outputs = {
            "Time-MoE":           output_time_moe_200,
            "Timer":              output_timer,
            "TimesFM":            output_timesfm,
            "Moirai-Small":       output_moirai_small,
            "Chronos-Bolt-Small": output_chronos_bolt_small,
            "FM-MoE":             output_mymoe,
            "AutoETS":            output_autoets,
            "AutoARIMA":          output_autoarima,
            "SeasonalNaive":      output_seasonal_naive,
            "NBEATS":             output_nbeats,
            "RandomForest":       output_rf,
            "XGBRegressor":       output_xgb,
            "PatchTST":           output_patchtst,
            "iTransformer":       output_itransformer,
        }

        ensemble_models  = {k: v for k, v in model_outputs.items() if k != "FM-MoE"}
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
                mape_list.append(round(mape, 4))
            results[name] = mape_list

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.rename(columns={"index": "Modelo"}, inplace=True)
        df_results.columns = ["Modelo"] + list(range(n_series))

        df_times = pd.DataFrame(
            list(times_dict.items()),
            columns=["Modelo", "Tempo (s)"],
        )

        return df_results, df_times

    # ======================================
    # MAIN LOOP OVER BENCHMARK DATASETS
    # ======================================
    for dataset_name, cfg in DATASETS.items():
        results_file = os.path.join(results_root, f"results_{dataset_name}.csv")
        times_file   = os.path.join(times_root,   f"times_{dataset_name}.csv")

        if os.path.isfile(results_file) and os.path.isfile(times_file):
            print(f"[SKIP] {dataset_name} — results already exist", flush=True)
            continue

        print(f"\n===== Processing: {dataset_name} =====", flush=True)

        input_list, target_list = load_benchmark_dataset(dataset_name, cfg)
        if input_list is None:
            print(f"[SKIP] {dataset_name} — no valid data", flush=True)
            continue

        try:
            df_results, df_times = process_dataset(dataset_name, cfg, input_list, target_list)
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}", flush=True)
            continue

        if df_results is None or df_times is None:
            print(f"[SKIP] {dataset_name} — process_dataset returned None", flush=True)
            continue

        df_results.to_csv(results_file, index=False)
        df_times.to_csv(times_file, index=False)
        print(f"[DONE] {dataset_name}", flush=True)

    print("\nAll benchmark processing completed.", flush=True)


# ======================================
# ENTRY POINT
# ======================================
if __name__ == "__main__":
    run_full_experiment_pipeline(
        path_trained_models="trained_models_benchmark/topk_2_norm_std_noise_True_ep_30_lr_0.0001",
        experiment_name="model_topk_2_norm_std_noise_True_ep_30_lr_0.0001",
        top_k=2,
        use_noise=True,
    )