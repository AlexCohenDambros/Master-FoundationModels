import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimerExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        
    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True

        )
        model.to(self.device)
        model.eval()
        
        input_tensor = input_tensor.to(self.device)

        outputs = [] 

        with torch.no_grad():
            for i in range(input_tensor.size(0)):      
                past_target =  input_tensor[i].unsqueeze(0)

                forecast = model.generate(
                    past_target,
                    max_new_tokens=prediction_length,
                    num_samples = 100
                )

                out_row = torch.as_tensor(forecast.mean(dim=1), dtype=torch.float32).reshape(1, -1).to(self.device)
                outputs.append(out_row)

        out = torch.cat(outputs, dim=0)

        return out