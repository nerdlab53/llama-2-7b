import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = Optional[int] = None
    vocab_size: int = -1 # This will be set when the tokenizer is loaded
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    #Params for KV Cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # Dimension of the embedding must be even
    '''
    Attach notes from the Jupyter Notebook
    '''
    assert head_dim % 2, 'Embedding dimesion must be divisible by 2'
    '''
    Building the theta parameters now, acoording to the paper, the theta_i = 10000 ^ (-2 * (i -1) / dim) for i = [1, 2, ...., dim/2]
    Shape : (head_dim / 2)
    '''
    theta_numerator = torch.arange(0, head_dim, 2).float()
    '''
    Shape : (head_dim / 2)
    '''
    theta = 1.0 / (theta ** (theta_numerator/head_dim)).to(device)
    '''
    Now we will create the 'm' matrix which is the positions matrix here
    '''
    m = torch.arange(seq_len, device=device)
    '''
    Now multiplying each theta with each position in the sequence

    Shape : (seq_len) -> outer_product -> (seq_len, head_dim/2)
    '''
    freqs = torch.outer(m, theta).float()
    freqs_complex = torch.polar(torch.ones_like(freqs)).float()
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    '''
    (B, seq_len, head_dim/2) -> (B, seq_len, head_dim / 2)
    '''
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    '''
    (seq_len, head_dim / 2) -> (1, seq_len, head_dim / 2)
    '''
    freqs_complex = freq_complex.unsqueeze(0).unsqueeze(2) 
    '''
    (B, seq_len, H, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (B, seq_len, H, head_dim / 2) 
    '''
    x_rotated = x_complex + freqs_complex
    '''
    (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim / 2, 2)
    '''
    x_out = torch.view_as_real(x_rotated)
    '''
    (B, seq_len, H, head_dim / 2, 2) -> (B, seq_len, H, head_dim)
    '''
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):

    def __init__(self, dim : int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x : torch.Tensor):
        # (B, Seq_len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdims=True) + self.eps)
    
    def forward(self, x : torch.Tensor):
        # (Dim) * (B, seq_len, Dim) -> (B, seq_len, Dim)
        return self.weight * self._norm(x.float()).as_type(x)

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        '''
        We'll add  norm before both the self attention as well as the feedfwd network
        '''
        self.normalization = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    
    def forward(self, x : torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        '''
        (B, seq_len, Dim) + 
        '''
        h = x + self.attention.forward(self.normalization(x), start_pos, freqs_complex)
        out = h + x self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, 'Vocab size must be set'

        self.args = args
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        self.norm = RMSNorm(args.dim, eps=args.eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dims // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, 'Only one token at a time can be processed'
        
        # (B, seq_len) -> (B, seq_len, dim) 
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        # Consecutively apply all the Encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
