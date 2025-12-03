#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import torch
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
import timesfm
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch

print("GPUs visíveis:", torch.cuda.device_count())
print("Usando:", torch.cuda.get_device_name(0))


# ### Get data SP

# In[2]:


file_path = "../dataset_global/dataset_global.jsonl"

data_list = []

train_list = []
test_list = []

with open(file_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        for key, value in entry.items():
            if key.endswith("_sp"):
                train_list.append(value[:-12])
                test_list.append(value[-12:])

tensor_train = torch.tensor(train_list, dtype=torch.float32)
tensor_test = torch.tensor(test_list, dtype=torch.float32)

print("Train shape:", tensor_train.shape)
print("Test shape:", tensor_test.shape)


# In[3]:


device = "cuda"
if device.lower() == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

print("Device: ", device)


# ### StandarScaler

# In[4]:


# Guardar média e desvio padrão de cada série para desnormalização depois
mean_vals = tensor_train.mean(dim=1, keepdim=True)
std_vals = tensor_train.std(dim=1, keepdim=True)

# Evitar divisão por zero
std_vals[std_vals == 0] = 1e-8

# Normalizar cada série (linha)
tensor_train_scaled = (tensor_train - mean_vals) / std_vals


# In[5]:


prediction_length = 12
context_length = 398


# ## Time-Moe

# In[ ]:


# model = AutoModelForCausalLM.from_pretrained(
#     "Maple728/TimeMoE-200M",
#     trust_remote_code=True,
#     device_map=device,
# )

# input_timemoe = tensor_train_scaled

# output = model.generate(input_timemoe, max_new_tokens=prediction_length) 
# output_time_moe_scaled  = output[:, -prediction_length:]

# output_time_moe = output_time_moe_scaled * std_vals + mean_vals

# print("TimeMoE")


# In[ ]:


# mape_series = (torch.abs((tensor_test - output_time_moe) / tensor_test)).mean(dim=1) * 100

# for i, mape_val in enumerate(mape_series):
#     print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## Timer

# In[6]:


model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            device_map=device,

        )


model.to(device)
model.eval()

input_tensor = tensor_train_scaled.to(device)

with torch.no_grad():

    forecast = model.generate(
        input_tensor,
        max_new_tokens=prediction_length,
    )

output_timer_scaled = forecast.squeeze(1)


# output_timer = output_timer_scaled * (max_vals - min_vals) + min_vals
output_timer = output_timer_scaled * std_vals + mean_vals

print("Timer")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_timer) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## TimesFM

# In[ ]:


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

input_np = tensor_train_scaled.clone().detach().cpu().numpy()

with torch.no_grad():
    out, experimental_quantile_forecast = model.forecast(input_np) 

output_timesfm_scaled = torch.from_numpy(out).float()

# output_timesfm = output_timesfm_scaled * (max_vals - min_vals) + min_vals
output_timesfm = output_timesfm_scaled * std_vals + mean_vals

print("TimesFM")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_timesfm) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## Chronos

# In[ ]:


from chronos import BaseChronosPipeline

model = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map=device,
    torch_dtype=torch.bfloat16,
)

with torch.no_grad():
    _, output_chronos_scaled = model.predict_quantiles(
        context=tensor_train_scaled,
        prediction_length=prediction_length,
    )

# output_chronos = output_chronos_scaled * (max_vals - min_vals) + min_vals
output_chronos = output_chronos_scaled * std_vals + mean_vals

print("Chronos")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_chronos) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_chronos) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## Moirai

# In[ ]:


from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

outputs = [] 

for i in range(tensor_train_scaled.size(0)):      
    past_target = tensor_train_scaled[i].unsqueeze(0).unsqueeze(-1)

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

# output_moirai = output_moirai_scaled * (max_vals - min_vals) + min_vals
output_moirai = output_moirai_scaled * std_vals + mean_vals

print("Moirai")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_moirai) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## Moirai-MoE

# In[ ]:


model = MoiraiForecast(
            # TODO: check what other sizes are available on moirai-moe
            module=MoiraiModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=16,
            num_samples=20,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

outputs = [] 

for i in range(tensor_train_scaled.size(0)):      
    past_target = tensor_train_scaled[i].unsqueeze(0).unsqueeze(-1)

    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)  
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)  

    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )

    out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1)
    outputs.append(out_row)

output_moiraimoe_scaled = torch.cat(outputs, dim=0)

# output_moirai_moe = output_moiraimoe_scaled * (max_vals - min_vals) + min_vals
output_moirai_moe = output_moiraimoe_scaled * std_vals + mean_vals

print("MoiraiMoe")


# In[ ]:


mape_series = (torch.abs((tensor_test - output_moirai_moe) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## My MoE

# In[ ]:


from setup.models.modeling_model import predict_from_model

output_scaled = predict_from_model(model_path="moe_model.pt", series=tensor_train_scaled, horizon=prediction_length, context_length=398, device="cpu")

# output = output_scaled * (max_vals - min_vals) + min_vals
output = output_scaled * std_vals + mean_vals


# In[ ]:


mape_series = (torch.abs((tensor_test - output) / tensor_test)).mean(dim=1) * 100

for i, mape_val in enumerate(mape_series):
    print(f"Série {i}: MAPE = {mape_val.item():.2f}%")


# ## Summary
# 

# In[ ]:


model_outputs = {
    # "Moirai-MoE": output_moirai_moe,
    "Moirai": output_moirai,
    "Chronos": output_chronos,
    "TimesFM": output_timesfm,
    "Timer": output_timer,
    "Time-MoE": output_time_moe,
    "Meu-MoE": output
}

foundation_keys = [
    "Moirai-MoE",
    "Moirai",
    "Chronos",
    "TimesFM",
    "Timer",
    "Time-MoE"
]

foundation_mean = torch.stack([model_outputs[k] for k in foundation_keys], dim=0).mean(dim=0)
model_outputs["Média Foundation Models"] = foundation_mean

results = {}
for model_name, preds in model_outputs.items():
    mape_series = (torch.abs((tensor_test - preds) / tensor_test)).mean(dim=1) * 100
    results[model_name] = [round(val.item(), 2) for val in mape_series]

df_results = pd.DataFrame(results).T
df_results.columns = [f"Série {i+1}" for i in range(df_results.shape[1])]

df_results

