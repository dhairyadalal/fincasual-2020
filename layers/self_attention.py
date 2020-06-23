import math

import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
        Original Implementation of the gelu activation function in Google Bert repo.
        https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))

def get_batch_sent_lens(batch: torch.tensor) -> torch.tensor:
    
    batch = batch.detach().cpu()
    lens = [len(np.where(row>0)[0]) for row in batch]
    return torch.tensor(lens)

class SelfAttention(nn.Module):
    
    def __init__(self,
                 attention_size: int,
                 batch_first: bool=False,
                 non_linearity: str="tanh"):
        super(SelfAttention, self).__init__()
        
        self.batch_first = batch_first
        self.attention_weights = nn.Parameter(FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)
        
        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        elif non_linearity == "gelu":
            self.non_linearity = gelu
        else:
            self.non_linearity = nn.Tanh()
        
        nn.init.uniform_(self.attention_weights, -0.005, 0.005)
        
    def get_mask(self,
                 attentions: torch.Tensor,
                 lengths: torch.Tensor) -> torch.Tensor:
        max_len = max(lengths)
        mask = torch.ones(attentions.size()).detach()
        
        if attentions.is_cuda:
            mask = mask.cuda()
            
        for i,l in enumerate(lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask
    
    def forward(self,
                inputs: torch.Tensor,
                lengths: torch.Tensor):
        # 1. Perform dot product of attention vector w/ each hidden state
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)
        
        # 2. Masking
        mask = self.get_mask(scores, lengths)
        masked_scores = scores * mask
        _sums = masked_scores.sum(-1, keepdim=True)
        scores = masked_scores.div(_sums)
        
        # 3. Weighted sums of hidden states by attention scores
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        attention = weighted.sum(1).squeeze()
        
        return attention, scores