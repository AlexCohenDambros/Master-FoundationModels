import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimerExpert(nn.Module):
    def __init__(self, prediction_length: int, context_length: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m",
            trust_remote_code=True
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                input_tensor,
                max_new_tokens=self.prediction_length,
                num_samples = 100
            ).squeeze(0).mean(dim=1)

        return out