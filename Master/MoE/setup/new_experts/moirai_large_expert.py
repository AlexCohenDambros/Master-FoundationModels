import torch
import torch.nn as nn
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

class MoiraiLargeExpert(nn.Module):
    def __init__(self,  device: str = 'cpu'):
        super().__init__()
        self.device = device

    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-large"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size=16,
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        model.to(self.device)
        model.eval()  
        
        input_tensor = input_tensor.to(self.device)

        outputs = [] 

        with torch.no_grad():
            for i in range(input_tensor.size(0)):      
                past_target =  input_tensor[i].unsqueeze(0).unsqueeze(-1)

                past_observed_target = torch.ones_like(past_target, dtype=torch.bool)  
                past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)  

                forecast = model(
                    past_target=past_target,
                    past_observed_target=past_observed_target,
                    past_is_pad=past_is_pad,
                )

                out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1).to(self.device)
                outputs.append(out_row)

        out = torch.cat(outputs, dim=0)

        return out