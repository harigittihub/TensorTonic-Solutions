import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        Returns: None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        """
        Returns: convolved output tensor of shape (batch, out_channels, H-k+1, W-k+1)
        """
        batch, _, h, w = x.shape
        k = self.kernel_size
        h_out = h - k + 1
        w_out = w - k + 1
        out_ch = self.weight.shape[0]

        weight_flat = self.weight.reshape(out_ch, -1)
        output = torch.zeros(batch, out_ch, h_out, w_out)

        for i in range(h_out):
            for j in range(w_out):
                patch = x[:, :, i:i+k, j:j+k].reshape(batch, -1)
                output[:, :, i, j] = patch @ weight_flat.t() + self.bias

        return output
