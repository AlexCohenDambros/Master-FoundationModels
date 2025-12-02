import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimerExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()

        if device.lower() == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True,
            device_map=self.device,

        )
        model.to(self.device)
        model.eval()
        
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
    
            forecast = model.generate(
                input_tensor,
                max_new_tokens=prediction_length,
            )

        return forecast.squeeze(1)