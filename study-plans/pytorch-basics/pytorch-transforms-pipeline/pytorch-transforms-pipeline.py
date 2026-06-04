import torch

class TransformPipeline:
    """Returns: float32 tensor of shape (C, H, W) from __call__"""
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image):
        x = image.float() / 255.0
        x = x.permute(2, 0, 1)
        x = (x - self.mean) / self.std
        return x
