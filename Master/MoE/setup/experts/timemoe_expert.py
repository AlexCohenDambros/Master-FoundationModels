import torch
from transformers import AutoModelForCausalLM

class TimeMoEExpert():
    def __init__(self, prediction_length: int, device: str = 'cpu'):
        self.prediction_length = prediction_length
        self.device = device
        self.model = None

        self.model = AutoModelForCausalLM.from_pretrained(
            'Maple728/TimeMoE-50M',
            device_map=device,
            trust_remote_code=True
        )
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.device = "cuda:0" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        input_moe = input_tensor.clone().to(self.device)

        print("Time-MoE")
        with torch.no_grad():
            out = self.model.generate(input_moe, max_new_tokens=self.prediction_length)

        return out[:, -self.prediction_length:]
