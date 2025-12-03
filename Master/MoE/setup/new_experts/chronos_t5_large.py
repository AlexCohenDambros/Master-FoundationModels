import torch
import torch.nn as nn
from chronos import BaseChronosPipeline

class Chronost5LargeExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()

        if device.lower() == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        input_tensor = input_tensor.to(self.device)

        model = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

        with torch.no_grad():
            _, output_chronos_scaled = model.predict_quantiles(
                context=input_tensor,
                prediction_length=prediction_length,
            )

        return output_chronos_scaled.to(self.device)