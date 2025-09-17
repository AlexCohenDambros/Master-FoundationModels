import torch
import torch.nn as nn
from chronos import BaseChronosPipeline

class ChronosExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        with torch.no_grad():
            _, out_mean = model.predict_quantiles(
                context=input_tensor,
                prediction_length=input_tensor[1],
            )

        return out_mean