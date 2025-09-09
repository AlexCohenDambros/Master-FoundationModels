import torch
import torch.nn as nn
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class MoiraiMoEExpert(nn.Module):
    def __init__(self, prediction_length: int, context_length: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = device

        self.model = MoiraiForecast(
            # TODO: check what other sizes are available on moirai-moe
            module=MoiraiModule.from_pretrained("Salesforce/moirai-moe-1.0-R-small"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        self.model.to(device)
        self.model.eval()  

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        outputs = [] 

        with torch.no_grad():
            for i in range(input_tensor.size(0)):      
                past_target =  input_tensor[i].unsqueeze(0).unsqueeze(-1)

                past_observed_target = torch.ones_like(past_target, dtype=torch.bool)  
                past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)  

                forecast = self.model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                )

                out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1).to(self.device)
                outputs.append(out_row)

        out = torch.cat(outputs, dim=0)

        return out