import torch
import torch.nn as nn
from chronos import BaseChronosPipeline

class ChronosExpert(nn.Module):
    def __init__(self, prediction_length: int, context_length: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = device

        self.model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            _, out_mean = self.model.predict_quantiles(
                context=input_tensor,
                prediction_length=self.prediction_length,
            )

        return out_mean