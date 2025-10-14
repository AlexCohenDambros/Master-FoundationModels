import os
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM
from chronos import BaseChronosPipeline
import timesfm
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from setup.models.modeling_model import predict_from_model

# ========================
# ==== CONFIG ============
# ========================

file_path = "../dataset_global/dataset_global.jsonl"
prediction_length = 12
results_path = "results_by_state_year"
os.makedirs(results_path, exist_ok=True)

context_by_year = {
    2024: 398,
    2023: 386,
    2022: 374,
    2021: 362,
    2020: 350
}

base_path = "../all_datasets_global_by_years"  # para iterar pelos estados


# ========================
# ==== PROCESS FUNCTION ===
# ========================

def process_dataset(state_code, year, context_length):
    """Carrega, filtra e processa o dataset global para um estado e ano específico."""

    train_list, test_list = [], []

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            for key, value in entry.items():
                if key.endswith(f"_{state_code}"):
                    total_len = context_length + prediction_length
                    if len(value) >= total_len:
                        seq = value[:total_len]

                        # print(seq)
                
                        train_list.append(seq[:-prediction_length])
                        test_list.append(seq[-prediction_length:])

    if len(train_list) == 0:
        print(f"No valid sequences found for state={state_code}, year={year}")
        return None

    tensor_train = torch.tensor(train_list, dtype=torch.float32)
    tensor_test = torch.tensor(test_list, dtype=torch.float32)

    # Normalize
    mean_vals = tensor_train.mean(dim=1, keepdim=True)
    std_vals = tensor_train.std(dim=1, keepdim=True)
    std_vals[std_vals == 0] = 1e-8
    tensor_train_scaled = (tensor_train - mean_vals) / std_vals

    # ========================
    # ==== TIME-MoE ==========
    # ========================
    model = AutoModelForCausalLM.from_pretrained(
        "Maple728/TimeMoE-200M",
        trust_remote_code=True,
    )
    input_timemoe = tensor_train_scaled.clone().detach()
    output = model.generate(input_timemoe, max_new_tokens=prediction_length)
    output_time_moe_scaled = output[:, -prediction_length:]
    output_time_moe = output_time_moe_scaled * std_vals + mean_vals

    # ========================
    # ==== TIMER =============
    # ========================
    model = AutoModelForCausalLM.from_pretrained(
        "thuml/sundial-base-128m",
        trust_remote_code=True,
    )
    outputs = []
    input_timer = tensor_train_scaled.clone().detach()
    for i in range(input_timer.size(0)):
        past_target = input_timer[i].unsqueeze(0)
        forecast = model.generate(
            past_target,
            max_new_tokens=prediction_length,
            num_samples=20
        )
        out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1)
        outputs.append(out_row)
    output_timer_scaled = torch.cat(outputs, dim=0)
    output_timer = output_timer_scaled * std_vals + mean_vals

    # ========================
    # ==== TimesFM ===========
    # ========================
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
    input_timesfm = tensor_train_scaled.clone().detach().cpu().numpy()
    with torch.no_grad():
        out, _ = model.forecast(input_timesfm)
    output_timesfm_scaled = torch.from_numpy(out).float()
    output_timesfm = output_timesfm_scaled * std_vals + mean_vals

    # ========================
    # ==== CHRONOS ===========
    # ========================
    model = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cpu",
        torch_dtype=torch.bfloat16,
    )
    input_chronos = tensor_train_scaled.clone().detach()
    with torch.no_grad():
        _, output_chronos_scaled = model.predict_quantiles(
            context=input_chronos,
            prediction_length=prediction_length,
        )
    output_chronos = output_chronos_scaled * std_vals + mean_vals

    # ========================
    # ==== MOIRAI ============
    # ========================
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
    outputs = []
    input_moirai = tensor_train_scaled.clone().detach()
    for i in range(input_moirai.size(0)):
        past_target = input_moirai[i].unsqueeze(0).unsqueeze(-1)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)
        forecast = model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1)
        outputs.append(out_row)
    output_moirai_scaled = torch.cat(outputs, dim=0)
    output_moirai = output_moirai_scaled * std_vals + mean_vals

    # ========================
    # ==== MY-MoE ============
    # ========================
    model_path = f"trained_models/excluding_{state_code}/model_excluding_{state_code}_{year}.pt"
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found: {model_path}. Skipping My-MoE for this dataset.")
        output_my = torch.zeros_like(output_timer)
    else:
        input_my = tensor_train_scaled.clone().detach()
        output_scaled = predict_from_model(
            model_path=model_path,
            series=input_my,
            horizon=prediction_length,
            context_length=context_length,
            device="cpu"
        )
        output_my = output_scaled * std_vals + mean_vals

    # ========================
    # ==== SUMMARY ===========
    # ========================
    model_outputs = {
        "Moirai": output_moirai,
        "Chronos": output_chronos,
        "TimesFM": output_timesfm,
        "Timer": output_timer,
        "Time-MoE": output_time_moe,
        "My-MoE": output_my
    }

    foundation_keys = ["Moirai", "Chronos", "TimesFM", "Timer", "Time-MoE"]
    foundation_mean = torch.stack([model_outputs[k] for k in foundation_keys], dim=0).mean(dim=0)
    model_outputs["Mean Foundation Models"] = foundation_mean

    results = {}
    for model_name, preds in model_outputs.items():
        # Ensure no negative predictions
        preds = preds.clone()  # make a copy to avoid modifying the original tensor
        preds[preds < 0] = 0

        # TODO: develop MAPE without exploding the results
        mape_series = (torch.abs((tensor_test - preds) / tensor_test)).mean(dim=1) * 100
        results[model_name] = [round(val.item(), 4) for val in mape_series]

    df_results = pd.DataFrame(results).T
    df_results.columns = [f"Series {i+1}" for i in range(df_results.shape[1])]
    return df_results


# ========================
# ==== MAIN LOOP =========
# ========================
all_results = {}

for state_folder in sorted(os.listdir(base_path)):
    state_path = os.path.join(base_path, state_folder)
    if not os.path.isdir(state_path):
        continue

    state_code = state_folder.replace("excluding_", "")

    for year in context_by_year.keys():
        context_length = context_by_year[year]

        print(f"Processing {state_code.upper()} - {year} (context_length={context_length})")

        df_results = process_dataset(state_code, year, context_length)
        if df_results is None:
            continue

        all_results[(state_code, year)] = df_results

        out_path = os.path.join(results_path, f"results_{state_code}_{year}.csv")
        df_results.to_csv(out_path)

# ========================
# ==== GLOBAL SUMMARY ====
# ========================
summary = {}
for (state, year), df in all_results.items():
    mean_per_model = df.mean(axis=1)
    summary[(state, year)] = mean_per_model

df_summary = pd.DataFrame(summary).T
df_summary.to_csv("results_summary.csv")

print("Processing complete. All summaries saved.")