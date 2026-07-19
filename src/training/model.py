"""
model.py — Dual-head chess transformer.

Architecture:
  1. CNN encoder:     (17, 8, 8) board tensor → 256-dim embedding per move
  2. Time projection: concat time features (3,) → Linear(259, 256)
  3. CLS token:       prepended to sequence for game-level Elo prediction
  4. Transformer:     4 layers, 4 heads, 256-dim, FFN=1024
  5. Head 1:          per-move tokens → Linear(256, 1) → cp_loss
  6. Head 2:          CLS token → Linear(256, 2) → (white_elo, black_elo)

Input shapes (batch first):
  board_tensors:  (B, T, 17, 8, 8)
  time_features:  (B, T, 3)
  attention_mask: (B, T)       1=real, 0=padding

Output:
  cp_loss_pred:   (B, T)       centipawn loss per move
  elo_pred:       (B, 2)       (white_elo, black_elo) per game
"""

import torch
import torch.nn as nn
import math


# ── CNN Encoder ───────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Encodes an 8x8x17 board tensor into a 256-dim embedding.

    3 conv layers with batch norm and ReLU:
      Layer 1: 17 → 64  channels, 3x3 kernel, padding=1 (keeps 8x8)
      Layer 2: 64 → 128 channels, 3x3 kernel, padding=1
      Layer 3: 128 → 256 channels, 3x3 kernel, padding=1

    Global average pooling: (256, 8, 8) → (256,)
    Final Linear: 256 → 256 (projection)

    Processes each move independently — same weights for all moves in sequence.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(17, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layer 3
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
        )

        # Global average pool: (d_model, 8, 8) → (d_model,)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 17, 8, 8) — batch of game sequences
        Returns:
            (B, T, d_model) — one embedding per move
        """
        B, T, C, H, W = x.shape

        # Flatten batch and time dims to process all moves in parallel
        x = x.view(B * T, C, H, W)       # (B*T, 17, 8, 8)
        x = self.conv(x)                   # (B*T, 256, 8, 8)
        x = self.pool(x)                   # (B*T, 256, 1, 1)
        x = x.view(B * T, -1)             # (B*T, 256)
        x = x.view(B, T, -1)              # (B, T, 256)
        return x


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Adds position information to move sequence so transformer
    knows move 1 vs move 40 vs move 100.
    """

    def __init__(self, d_model: int, max_len: int = 130, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe     = torch.zeros(max_len, d_model)
        pos    = torch.arange(0, max_len).unsqueeze(1).float()
        div    = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── Dual-Head Chess Transformer ───────────────────────────────────────────────

class ChessTransformer(nn.Module):
    """
    Dual-head transformer for chess move quality + Elo prediction.

    Head 1 (move quality): predicts centipawn loss per move
    Head 2 (Elo):          predicts white_elo + black_elo from CLS token
    """

    def __init__(
        self,
        d_model:     int   = 256,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        d_ff:        int   = 1024,
        n_time_feats: int  = 3,
        dropout:     float = 0.1,
        max_seq_len: int   = 129,  # 128 moves + 1 CLS token
    ):
        super().__init__()

        self.d_model = d_model

        # CNN encodes each board position → 256-dim
        self.cnn = CNNEncoder(d_model)

        # Project CNN output + time features → d_model
        # CNN gives 256-dim, time features give n_time_feats scalars
        # concat → Linear(256 + n_time_feats → 256)
        self.input_proj = nn.Linear(d_model + n_time_feats, d_model)

        # Learnable CLS token — prepended to sequence for Elo prediction
        # Shape: (1, 1, d_model) — broadcast across batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding over move sequence (including CLS at position 0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,   # (B, T, d_model) convention
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Elo head: game-level prediction from CLS token
        # predicts (white_elo, black_elo) — two scalars
        self.elo_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        board_tensors:  torch.Tensor,   # (B, T, 17, 8, 8)
        time_features:  torch.Tensor,   # (B, T, n_time_feats)
        attention_mask: torch.Tensor,   # (B, T) — 1=real, 0=padding
    ) -> torch.Tensor:
        """
        Returns:
            elo_pred: (B, 2) — (white_elo, black_elo)
        """
        B, T = board_tensors.shape[:2]

        # 1. Encode each board position with CNN → (B, T, 256)
        board_emb = self.cnn(board_tensors)

        # 2. Concat time features and project → (B, T, 256)
        x = torch.cat([board_emb, time_features], dim=-1)  # (B, T, 256+3)
        x = self.input_proj(x)                              # (B, T, 256)

        # 3. Prepend CLS token → (B, T+1, 256)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, 256)
        x   = torch.cat([cls, x], dim=1)        # (B, T+1, 256)

        # 4. Add positional encoding
        x = self.pos_enc(x)

        # 5. Build attention mask for transformer
        # CLS token is always real (prepend 1 to each mask row)
        cls_mask = torch.ones(B, 1, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, T+1)

        # PyTorch transformer uses True=ignore, False=attend (opposite convention)
        src_key_padding_mask = (full_mask == 0)  # (B, T+1)

        # 6. Transformer encoder
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, T+1, 256)

        # 7. Split CLS token and move tokens
        cls_out  = x[:, 0, :]    # (B, 256)      — game-level representation
        move_out = x[:, 1:, :]   # (B, T, 256)   — per-move representations

        # Elo from CLS token
        elo_pred = self.elo_head(cls_out)  # (B, 2)

        return elo_pred


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 4, 60  # batch of 4 games, 60 moves each

    model = ChessTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    board_tensors  = torch.randn(B, T, 17, 8, 8)
    time_features  = torch.randn(B, T, 3)
    attention_mask = torch.ones(B, T)

    elo_pred = model(board_tensors, time_features, attention_mask)

    print(f"elo_pred: {tuple(elo_pred.shape)}")   # (4, 2)
    print("Model forward pass OK")