import torch.nn as nn
import copy

# -------------------
# EarlyStopping
# -------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_loss = float("inf")
        self.best_model_state = None

    def __call__(self, loss: float, model: nn.Module):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.best_loss = loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_loss = loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def load_best_model(self, model: nn.Module):
        model.load_state_dict(self.best_model_state)