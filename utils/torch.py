import torch
import numpy as np

class TorchHelper:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def f(self, x):
        if isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = np.array(x)  # Efficient conversion
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def i(self, x):
        if isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = np.array(x)
        return torch.tensor(x, dtype=torch.int32, device=self.device)

    def l(self, x):
        if isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = np.array(x)
        return torch.tensor(x, dtype=torch.int64, device=self.device)
