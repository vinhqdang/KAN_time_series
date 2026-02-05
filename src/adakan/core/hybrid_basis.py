import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBasisFunction(nn.Module):
    def __init__(self, wavelet_levels=4, spline_grid=5, spline_order=3):
        super().__init__()
        self.wavelet_levels = wavelet_levels
        self.spline_grid = spline_grid
        self.spline_order = spline_order
        self.wavelet_coeffs = nn.Parameter(torch.randn(wavelet_levels + 1) * 0.1)
        self.spline_coeffs = nn.Parameter(torch.randn(spline_grid + spline_order) * 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def _eval_wavelet(self, x, coeffs):
        x_clamped = torch.clamp(x, 0.0, 1.0)
        output = coeffs[0]
        for level in range(1, len(coeffs)):
            scale = 2 ** (level - 1)
            for k in range(scale):
                left = k / scale
                right = (k + 0.5) / scale
                mid = (k + 1) / scale
                mask_pos = (x_clamped >= left) & (x_clamped < right)
                mask_neg = (x_clamped >= right) & (x_clamped < mid)
                output = output + coeffs[level] * (mask_pos.float() - mask_neg.float())
        return output

    def _eval_spline(self, x, coeffs, grid, order):
        grid_size = grid
        x_clamped = torch.clamp(x, 0.0, 1.0)
        idx = x_clamped * (grid_size - 1)
        k = idx.long()
        w1 = idx - k
        w0 = 1.0 - w1
        c0 = coeffs[torch.clamp(k, 0, len(coeffs) - 1)]
        c1 = coeffs[torch.clamp(k + 1, 0, len(coeffs) - 1)]
        return c0 * w0 + c1 * w1

    def forward(self, x):
        wavelet_out = self._eval_wavelet(x, self.wavelet_coeffs)
        spline_out = self._eval_spline(x, self.spline_coeffs, self.spline_grid, self.spline_order)
        return wavelet_out + torch.sigmoid(self.alpha) * spline_out
