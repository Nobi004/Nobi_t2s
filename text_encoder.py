import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=100, d_model=256, nhead=8, num_layers=4):
        super(TextEncoder, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=False  # Transformer expects [seq_len, batch, embed]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        x: [batch_size, seq_len] - tokenized input
        returns: [batch_size, seq_len, d_model] - encoded output
        """
        x = self.embedding(x)               # [batch, seq_len, d_model]
        x = x.permute(1, 0, 2)              # [seq_len, batch, d_model]
        x = self.transformer_encoder(x)     # [seq_len, batch, d_model]
        x = x.permute(1, 0, 2)              # [batch, seq_len, d_model]
        return x

# ---------------------
# 🔁 Example usage:
# ---------------------

# Parameters
vocab_size = 100
seq_len = 10
batch_size = 2
d_model = 256
nhead = 8

# Model
model = TextEncoder(vocab_size=vocab_size, d_model=d_model, nhead=nhead)

# Dummy input: batch of 2 sequences, each of length 10
sample_input = torch.randint(0, vocab_size, (batch_size, seq_len))

# Forward pass
output = model(sample_input)

print("Input shape:", sample_input.shape)   # [2, 10]
print("Output shape:", output.shape)        # [2, 10, 256]
