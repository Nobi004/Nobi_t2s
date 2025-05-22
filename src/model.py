import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, kernel_size=31, dropout=0.1):
        super().__init__()
        self.feedforward1 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size, padding=kernel_size//2, groups=d_model),
            nn.GLU(dim=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.feedforward2 = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + 0.5 * self.feedforward1(self.norm1(x))
        x = x + self.attention(self.norm2(x), self.norm2(x), self.norm2(x))[0]
        x = x + self.conv(self.norm3(x).permute(0, 2, 1)).permute(0, 2, 1)
        x = x + 0.5 * self.feedforward2(self.norm4(x))
        return x

class ConformerEncoder(nn.Module):
    def __init__(self, vocab_size=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]  # [batch, seq_len, d_model]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)  # [batch, seq_len, d_model]

class DurationPredictor(nn.Module):
    def __init__(self, in_dim=256, filter_size=256, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filter_size, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=1)
        self.linear = nn.Linear(filter_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(x.permute(0, 2, 1)).squeeze(-1)
        return x

class PitchPredictor(nn.Module):
    def __init__(self, in_dim=256, filter_size=256, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, filter_size, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(filter_size, filter_size, kernel_size, padding=1)
        self.linear = nn.Linear(filter_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.linear(x.permute(0, 2, 1)).squeeze(-1)
        return x

class VarianceAdaptor(nn.Module):
    def __init__(self, d_model=256, max_frames=80):
        super().__init__()
        self.duration_predictor = DurationPredictor(d_model)
        self.pitch_predictor = PitchPredictor(d_model)
        self.energy_predictor = PitchPredictor(d_model)
        self.pitch_proj = nn.Linear(1, d_model)
        self.energy_proj = nn.Linear(1, d_model)
        self.max_frames = max_frames

    def expand_hidden(self, H, D):
        batch_size, seq_len, d_model = H.size()
        D = D.long().clamp(min=0)
        max_T = min(D.sum(dim=1).max(), self.max_frames)
        H_expanded = torch.zeros(batch_size, max_T, d_model, device=H.device)
        for b in range(batch_size):
            indices = torch.repeat_interleave(torch.arange(seq_len, device=H.device), D[b])
            if len(indices) > max_T:
                indices = indices[:max_T]
            H_expanded[b, :len(indices)] = H[b, indices]
        return H_expanded

    def forward(self, H, D_gt=None, P_gt=None, E_gt=None, is_inference=False):
        D_pred = self.duration_predictor(H)
        D = D_pred if is_inference else D_gt
        H_expanded = self.expand_hidden(H, D)
        P_pred = self.pitch_predictor(H_expanded)
        E_pred = self.energy_predictor(H_expanded)
        P = P_pred if is_inference else P_gt
        E = E_pred if is_inference else E_gt
        P_proj = self.pitch_proj(P.unsqueeze(-1))
        E_proj = self.energy_proj(E.unsqueeze(-1))
        H_adapted = H_expanded + P_proj + E_proj
        return H_adapted, D_pred, P_pred, E_pred

class GatedDilatedConv(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation, padding=padding)

    def forward(self, x):
        z = self.conv(x)
        z1, z2 = z.chunk(2, dim=1)
        return torch.tanh(z1) * torch.sigmoid(z2)

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.gated_conv = GatedDilatedConv(channels, kernel_size, dilation)
        self.conv1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = self.gated_conv(x)
        residual = self.conv1x1(out)
        return x + residual

class WaveformDecoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=1, num_blocks=30, channels=64):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, channels, kernel_size=256, stride=256)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size=3, dilation=2**(i % 10))
            for i in range(num_blocks)
        ])
        self.final_conv = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upsample(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x.squeeze(1)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, channels=64):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else channels, channels, kernel_size=15, stride=2, padding=7),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ) for i in range(4)
        ])
        self.final = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 20480] -> [batch, 1, 20480]
        for layer in self.layers:
            x = layer(x)
        return self.final(x).squeeze(1)  # [batch, seq_len]

class EndToEndTTS(nn.Module):
    def __init__(self, vocab_size=100, d_model=256, nhead=4, num_encoder_layers=4, num_blocks=30, channels=64):
        super().__init__()
        self.encoder = ConformerEncoder(vocab_size, d_model, nhead, num_encoder_layers)
        self.variance_adaptor = VarianceAdaptor(d_model, max_frames=80)
        self.waveform_decoder = WaveformDecoder(d_model, num_blocks=num_blocks, channels=channels)
        self.discriminator = Discriminator()

    def forward(self, text, D_gt=None, P_gt=None, E_gt=None, is_inference=False):
        H = self.encoder(text)
        H_adapted, D_pred, P_pred, E_pred = self.variance_adaptor(H, D_gt, P_gt, E_gt, is_inference)
        H_adapted = H_adapted.permute(0, 2, 1)  # [batch, max_frames, d_model] -> [batch, d_model, max_frames]
        waveform = self.waveform_decoder(H_adapted)
        if is_inference:
            return waveform, D_pred, P_pred, E_pred
        disc_score = self.discriminator(waveform)
        return waveform, D_pred, P_pred, E_pred, disc_score