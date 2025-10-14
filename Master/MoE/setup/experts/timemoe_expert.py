import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class TimeMoEExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-200M",
            trust_remote_code=True,
        )
        model.to(self.device)
        model.eval()

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            out = model.generate(
                input_tensor,
                max_new_tokens=prediction_length
            )

        return out[:, -prediction_length:]