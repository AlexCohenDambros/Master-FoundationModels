import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimeMoEExpert(nn.Module):
    def __init__(self, prediction_length: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_length = prediction_length
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-50M",
            trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        print("Time-MoE")
        with torch.no_grad():
            out = self.model.generate(
                input_tensor,
                max_new_tokens=self.prediction_length
            )

        return out[:, -self.prediction_length:]