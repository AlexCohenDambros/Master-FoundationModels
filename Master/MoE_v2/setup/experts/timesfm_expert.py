import torch
import torch.nn as nn
import timesfm

class TimesFMExpert(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
    
    def forward(self, input_tensor: torch.Tensor, context_length: int, prediction_length: int) -> torch.Tensor:
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=self.device,
                per_core_batch_size=32,
                horizon_len=prediction_length,
                num_layers=50,
                use_positional_embedding=False,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
            ),
        )

        
        input_np = input_tensor.clone().detach().cpu().numpy()
        
        with torch.no_grad():
            out, experimental_quantile_forecast = model.forecast(input_np) 

        return torch.from_numpy(out).float()