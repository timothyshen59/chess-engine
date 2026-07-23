import torch
import torch.nn as nn 
import math 

class CNNEncoder(nn.Module): 
    """CNN with hierarchical deduplication"""
    
    def __init__(self, d_model: int = 256): 
        super().__init__() 
        
        self.conv = nn.Sequential(
            nn.Conv2d(17,64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.Conv2d(64,128, kernel_size=3, padding = 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Conv2d(128, d_model, kernel_size=3, padding = 1), 
            nn.BatchNorm2d(d_model), 
            nn.ReLU(), 
    
        )
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
    def _board_hash(self, x_flat: torch.Tensor)->torch.Tensor: 
        x_int = x_flat.to(torch.uint8).to(torch.int64) 
        
        prime  = torch.tensor(31, dtype=torch.int64, device=x_flat.device)
        powers = torch.pow(prime, torch.arange(x_int.shape[1], device=x_flat.device, dtype=torch.int64))
        hashes = (x_int * powers).sum(dim=1)  # (N,)
        return hashes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """        
        Args:
            x: (B, T, 17, 8, 8)
        Returns:
            (B, T, d_model)
        """
        B, T, C, H, W = x.shape 
        N = B * T 
        
        x_flat = x.view(N,C,H,W) 
        x_2d = x_flat.view(N, -1).float()
        
        hashes = self._board_hash(x_2d)
        
        _,inverse,counts= torch.unique(hashes, return_inverse=True, return_counts = True)
        
        unique_indices = torch.zeros(counts.shape[0], dtype = torch.long, device = x.device)
        
        idx = torch.arange(N-1,-1,-1, device=x.device)
        unique_indices.scatter_(0, inverse[idx], idx)
        
        unique_boards = x_flat[unique_indices]
        
        emb = self.conv(unique_boards)
        emb = self.pool(emb) 
        emb = emb.view(-1, emb.shape[1])
        
        x_emb = emb[inverse]
        return x_emb.view(B,T, -1)
    
class PositionalEncoding(nn.Module): 
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 130, dropout: float = 0.1): 
        super().__init__() 
        self.dropout = nn.Dropout(dropout) 
        
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])

class ChessTransformer(nn.Module): 
    """CNN + Transformer model for elo prediction"""
    def __init__(
        self,
        d_model:      int   = 256,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        d_ff:         int   = 1024,
        n_time_feats: int   = 3,
        dropout:      float = 0.1,
        max_seq_len:  int   = 129,
    ):
        super().__init__() 
        
        self.d_model = d_model 
        self.cnn        = CNNEncoder(d_model)
        self.input_proj = nn.Linear(d_model + n_time_feats, d_model)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
 
        self.elo_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )
    
    def forward(self, board_tensors: torch.Tensor, time_features: torch.Tensor,  attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = board_tensors.shape[:2]
 
        board_emb = self.cnn(board_tensors)
 
        x = self.input_proj(torch.cat([board_emb, time_features], dim=-1))
 
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
 
        x = self.pos_enc(x)
 
        cls_mask = torch.ones(B, 1, device=attention_mask.device)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)
        src_key_padding_mask = (full_mask == 0)
 
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
 
        return self.elo_head(x[:, 0, :])  # (B, 2)

if __name__ == "__main__":
    import time
 
    B, T  = 64, 128
    model = ChessTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
 
    board_tensors  = torch.zeros(B, T, 17, 8, 8)
    board_tensors[:, :10] = board_tensors[:, 0:1].expand(-1, 10, -1, -1, -1)  # first 10 moves identical
    time_features  = torch.randn(B, T, 3)
    attention_mask = torch.ones(B, T)
 
    model(board_tensors, time_features, attention_mask)
 
    t0 = time.time()
    for _ in range(5):
        elo_pred = model(board_tensors, time_features, attention_mask)
    print(f"Avg forward: {(time.time()-t0)/5:.3f}s")
    print(f"elo_pred: {tuple(elo_pred.shape)}")