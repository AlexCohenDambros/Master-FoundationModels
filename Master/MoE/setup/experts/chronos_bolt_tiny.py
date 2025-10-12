import torch
import torch.nn as nn
from chronos import BaseChronosPipeline

class ChronosBoltTinyExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device

    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-tiny",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        with torch.no_grad():
            _, out_mean = model.predict_quantiles(
                context=input_tensor,
                prediction_length=prediction_length,
            )

        return out_mean