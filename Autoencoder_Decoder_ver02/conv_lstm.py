"""
ConvLSTM module - Simple implementation for compatibility
"""
import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    """
    Simple ConvLSTM cell implementation
    For compatibility with models that import from conv_lstm
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutions for input and hidden states
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        """
        Args:
            input_tensor: (B, C, H, W) - input at current time step
            cur_state: tuple of (h_cur, c_cur) - current hidden and cell states
        Returns:
            h_next, c_next: next hidden and cell states
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, C+H, H, W)
        
        # Compute gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell and hidden states
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        """
        Initialize hidden and cell states
        Args:
            batch_size: batch size
            image_size: (H, W) tuple
        Returns:
            (h, c): tuple of hidden and cell states
        """
        height, width = image_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, 
                       device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        c = torch.zeros(batch_size, self.hidden_dim, height, width,
                       device=self.conv.weight.device, dtype=self.conv.weight.dtype)
        return (h, c)

