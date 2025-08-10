import torch
from einops import rearrange
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class MoiraiMoEExpert():
    def __init__(self, prediction_length: int, device: str = 'cpu'):
        self.prediction_length = prediction_length
        self.device = device
        self.model = None

        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
            prediction_length=self.prediction_length,
            context_length=128,
            patch_size=16,          
            num_samples=1,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        required_len = self.model.hparams.context_length 

        print("MoiraiMoE")
        
        input_padded = input_tensor.clone()[:, :required_len]

        input_1d = input_padded.squeeze(0) 
        past_target = rearrange(input_1d, "t -> 1 t 1")

        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )

        with torch.no_grad():
            out = torch.as_tensor(forecast, dtype=torch.float32).reshape(1, -1)

        return out