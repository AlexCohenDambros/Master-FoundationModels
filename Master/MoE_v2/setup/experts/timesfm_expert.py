import torch
import torch.nn as nn
import timesfm

class TimesFMExpert(nn.Module):
    def __init__(self, prediction_length: int, device: str = 'cpu'):
        super().__init__()
        self.prediction_length = prediction_length
        self.device = device
        
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self.device,
                per_core_batch_size=32,
                horizon_len=self.prediction_length,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_np = input_tensor.clone().detach().cpu().numpy()

        print("TimesFM")
        with torch.no_grad():
            out = self.model.forecast(input_np)

        return torch.from_numpy(out[0])