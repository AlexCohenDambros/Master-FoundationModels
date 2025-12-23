import os
import json
import torch
import pandas as pd
import time
from transformers import AutoModelForCausalLM
from chronos import BaseChronosPipeline
import timesfm
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from setup.models.modeling_model import predict_from_model
from sklearn.metrics import mean_absolute_percentage_error

# ===============================
# CONFIG
# ===============================

file_path = "../dataset_global/dataset_global.jsonl"

HORIZONS = [3, 6, 12, 24]
YEARS = [2024, 2023, 2022, 2021, 2020]

BASE_CONTEXT = 410

results_root = "results_by_state_year"
times_root = "times_by_state_year"

os.makedirs(results_root, exist_ok=True)
os.makedirs(times_root, exist_ok=True)

base_path = "../all_datasets_global_by_years"

# ===============================
# CONTEXT FUNCTION (DYNAMIC)
# ===============================

def get_context_length(year, horizon):
    return BASE_CONTEXT - horizon - ((2024 - year) * horizon)

# ===============================
# MAIN PROCESS FUNCTION
# ===============================

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

    tensor_train = torch.tensor(train_list, dtype=torch.float32)
    tensor_test = torch.tensor(test_list, dtype=torch.float32)

    mean_vals = tensor_train.mean(dim=1, keepdim=True)
    std_vals = tensor_train.std(dim=1, keepdim=True)
    std_vals[std_vals == 0] = 1e-8
    tensor_train_scaled = (tensor_train - mean_vals) / std_vals

    times_dict = {}

    # -------- Time-MoE 200--------
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-200M", trust_remote_code=True
    )
    out = model.generate(tensor_train_scaled, max_new_tokens=prediction_length)
    out = out[:, -prediction_length:]
    output_time_moe_200 = out * std_vals + mean_vals
    times_dict["Time-MoE200M"] = round(time.time() - start, 4)

    # -------- Timer --------
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "thuml/sundial-base-128m", trust_remote_code=True
    )
    outs = []
    for i in range(tensor_train_scaled.size(0)):
        fc = model.generate(tensor_train_scaled[i].unsqueeze(0), max_new_tokens=prediction_length, num_samples=20)
        outs.append(torch.as_tensor(fc.mean(dim=1)).reshape(1, -1))
    output_timer = torch.cat(outs) * std_vals + mean_vals
    times_dict["Timer"] = round(time.time() - start, 4)

    # -------- TimesFM --------
    start = time.time()
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="cpu",
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
    output_timesfm = torch.from_numpy(out).float() * std_vals + mean_vals
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
    outs = []
    for i in range(tensor_train_scaled.size(0)):
        past = tensor_train_scaled[i].unsqueeze(0).unsqueeze(-1)
        obs = torch.ones_like(past, dtype=torch.bool)
        pad = torch.zeros_like(past, dtype=torch.bool).squeeze(-1)
        fc = model(past_target=past, past_observed_target=obs, past_is_pad=pad)
        outs.append(torch.as_tensor(fc.mean(dim=1)).reshape(1, -1))
    output_moirai_small = torch.cat(outs) * std_vals + mean_vals
    times_dict["Moirai-Small"] = round(time.time() - start, 4)

    # -------- Moirai Small --------
    start = time.time()
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=16,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

    # -------- Chronos-Bolt-Small --------
    start = time.time()
    model = BaseChronosPipeline.from_pretrained(
       "amazon/chronos-bolt-small", device_map="cpu", torch_dtype=torch.bfloat16
    )
    _, out = model.predict_quantiles(
        context=tensor_train_scaled, prediction_length=prediction_length
    )
    output_chronos_bolt_small = out * std_vals + mean_vals
    times_dict["Chronos-Bolt-Small"] = round(time.time() - start, 4)

    # -------- My-MoE --------
    start = time.time()
    model_path = f"trained_models/horizon_{prediction_length}/excluding_{state_code}/model_excluding_{state_code}_{year}.pt"
    if not os.path.exists(model_path):
        output_mymoe = torch.zeros_like(output_timer)
    else:
        out = predict_from_model(
            model_path=model_path,
            series=tensor_train_scaled,
            horizon=prediction_length,
            context_length=context_length,
            device="cpu"
        )
        output_mymoe = out * std_vals + mean_vals
    times_dict["My-MoE"] = round(time.time() - start, 4)

    model_outputs = {
        "Moirai-Small": output_moirai_small,
        "Time-MoE200M": output_time_moe_200,
        "TimesFM": output_timesfm,
        "Timer": output_timer,
        "Chronos-Bolt-Small": output_chronos_bolt_small,
        "My-MoE": output_mymoe
    }

    foundation_keys = [
        "Moirai",
        "Time-MoE",
        "TimesFM",
        "Timer",
        "Chronos",
        ]
    
    model_outputs["Mean Foundation Models"] = torch.stack(
        [model_outputs[k] for k in foundation_keys], dim=0
    ).mean(dim=0)

    results = {}
    for name, preds in model_outputs.items():
        preds[preds < 0] = 0
        mape_list = []
        for i in range(preds.shape[0]):
            mape = mean_absolute_percentage_error(
                tensor_test[i].numpy(), preds[i].numpy()
            ) * 100
            mape_list.append(round(mape, 4))
        results[name] = mape_list

    df_results = pd.DataFrame(results).T.reset_index()
    df_results.rename(columns={"index": "Modelo"}, inplace=True)
    df_results.columns = ["Modelo"] + [p.capitalize() for p in product_names]

    df_times = pd.DataFrame(list(times_dict.items()), columns=["Modelo", "Tempo (s)"])

    return df_results, df_times

# ===============================
# MAIN LOOP
# ===============================

for horizon in HORIZONS:
    print(f"\n===== Horizon {horizon} =====")

    results_path = os.path.join(results_root, f"horizon_{horizon}")
    times_path = os.path.join(times_root, f"horizon_{horizon}")

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(times_path, exist_ok=True)

    for state_folder in sorted(os.listdir(base_path)):
        if not os.path.isdir(os.path.join(base_path, state_folder)):
            continue

        state_code = state_folder.replace("excluding_", "")

        for year in YEARS:
            context_length = get_context_length(year, horizon)

            print(f"Processing {state_code.upper()} - {year} - Horizon {horizon} - Context {context_length}")

            df_results, df_times = process_dataset(
                state_code, year, context_length, horizon
            )

            if df_results is None:
                continue

            df_results.to_csv(
                os.path.join(results_path, f"results_{state_code}_{year}.csv"),
                index=False
            )

            df_times.to_csv(
                os.path.join(times_path, f"times_{state_code}_{year}.csv"),
                index=False
            )

print("\nAll processing completed.")
