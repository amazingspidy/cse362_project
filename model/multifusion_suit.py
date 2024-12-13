# from torch import nn
# import torch

# class Multifusion(nn.Module):
#   def __init__(self):
#     super(Multifusion, self).__init__()
#     ##### TODO #####
#     self.device = "cuda" if torch.cuda.is_available() else "cpu"
#     self.encode_meta = nn.Linear(2, 512, bias=True)
#     self.ffn = nn.Linear(512 * 3 , 512)
    
#   def forward(self, image_embedding, text_embedding, prices, likes):
#     ##### TODO #####

#     batch_size = image_embedding.shape[0]      
#     meta_feature = torch.stack([prices, likes], dim=-1) # [B, 8, 2]
#     meta_embedding = self.encode_meta(meta_feature) # [B, 8, 512]
    
#     # 특징들을 concatenate
#     fused_feature = torch.cat((image_embedding, text_embedding, meta_embedding), dim=1)  # [B, 8, 3, 512]
#     fused_vec   = fused_feature.view(batch_size, 8,  -1)    # [B, 8, 512 * 3]
    
#     # FFN으로 임베딩크기 축소
#     out = self.ffn(fused_vec)  # [batch_size, 8, 512]
#     return out


from torch import nn
import torch
from torch.functional import F
import math

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

class DiffAttn(nn.Module):
  def __init__(self, device, emb_dim, head_n, head_dim, depth):
    super(DiffAttn, self).__init__()
    self.device = device
    self.head_n = head_n
    self.head_dim = head_dim
    self.scaling = head_dim ** -0.5

    self.to_q = nn.Linear(emb_dim, 2 * head_n * head_dim, bias=False)
    self.to_k = nn.Linear(emb_dim, 2 * head_n * head_dim, bias=False)
    self.to_v = nn.Linear(emb_dim, 2 * head_n * head_dim, bias=False)
    self.to_out = nn.Linear(2 * head_n * head_dim, emb_dim, bias=False)

    self.lambda_init = lambda_init_fn(depth)
    self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
    self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

    self.subln = RMSNorm(2 * head_dim, eps=1e-5, elementwise_affine=True)

  def forward(self, query):
    batch_size, target_len, emb_dim = query.size()
    emb_len = target_len
    q = self.to_q(query)
    k = self.to_k(query)
    v = self.to_v(query)

    q = q.view(batch_size, target_len, 2 * self.head_n, self.head_dim).transpose(1, 2)
    k = k.view(batch_size, emb_len, 2 * self.head_n, self.head_dim).transpose(1, 2)
    v = v.view(batch_size, emb_len, self.head_n, 2 * self.head_dim).transpose(1, 2)

    q *= self.scaling

    score = q @ k.transpose(-1,-2)
    score = torch.nan_to_num(score)

    # mask_mat = mask.unsqueeze(1).repeat(1,target_len,1)
    # mask_mat = mask_mat  * mask_mat.transpose(-1,-2)
    # mask_mat = mask_mat.unsqueeze(1).repeat(1,2*self.head_n,1,1)
    # mask_mat = torch.where(mask_mat == 0, float("-inf"), 0)
    # score = score + mask_mat

    score = F.softmax(score, dim=-1, dtype=torch.float32).type_as(score)
    # score = torch.nan_to_num(score)

    lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
    lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
    lambda_full = lambda_1 - lambda_2 + self.lambda_init
    score = score.view(batch_size, self.head_n, 2, target_len, target_len)
    score = score[:, :, 0] - lambda_full * score[:, :, 1]

    attn = score @ v
    attn = self.subln(attn)
    attn = attn * (1 - self.lambda_init)
    attn = attn.transpose(1,2).reshape(batch_size, target_len, self.head_n * 2 * self.head_dim)

    out = self.to_out(attn)
    
    return out
  
class FFN(nn.Module):
   def __init__(self, emb_dim):
      super(FFN,self).__init__()
      self.w1 = nn.Linear(emb_dim,emb_dim, bias=False)
      self.w2 = nn.Linear(emb_dim,emb_dim, bias=False)
      self.w3 = nn.Linear(emb_dim,emb_dim, bias=False)
      self.norm = RMSNorm(emb_dim, eps=1e-5, elementwise_affine=True)
    
   def forward(self, input):
      return self.norm(self.w2(F.silu(self.w1(input) * self.w3(input))))
  
class Block(nn.Module):
   def __init__(self, device):
      super(Block, self).__init__()
      emb_dim = 512
      self.attn = DiffAttn(device, emb_dim, 8, 32, 1)
      self.ffn = FFN(emb_dim)
    
   def forward(self, token):
    token = self.attn(token) + token
    token = self.ffn(token) + token
    return token

class DiffTransformer(nn.Module):
  def __init__(self, device):
    super(DiffTransformer, self).__init__()
    
    self.block = Block(device)
    self.predict = nn.Parameter(torch.zeros(512, dtype=torch.float32))
    self.num_layer = 4
    self.fcn = nn.Sequential(
       nn.Linear(512, 256),
       nn.SiLU(),
       nn.Linear(256, 256),
       nn.SiLU(),
       nn.Linear(256, 128),
       nn.SiLU(),
       nn.Linear(128, 128),
       nn.SiLU(),
       nn.Linear(128, 64),
       nn.SiLU(),
       nn.Linear(64,1)
    )

  def forward(self, emb):
    batch_size = emb.shape[0]
    token = self.predict.view(1,1,-1).repeat(batch_size, 1, 1)
    token = torch.cat([token, emb], dim=1)
    # mask = torch.cat([torch.tensor(1., device='cuda').view(1,1).repeat(batch_size, 1), mask], dim=1)
    for _ in range(self.num_layer):
      token = self.block(token)

    prob = self.fcn(token)
    return prob
  
class Multifusion(nn.Module):
  def __init__(self):
    super(Multifusion, self).__init__()
    ##### TODO #####
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.TWfusion = DiffTransformer(self.device)
    
  def forward(self, image_embedding, text_embedding):
    ##### TODO #####
    batch_size = image_embedding.shape[0]

    # mask = (1 - (image_embedding==0).to(torch.float32))[:,:,0]

    # emb_concat = torch.cat((text_embedding, image_embedding), dim=-1)
    emb_concat = image_embedding + text_embedding

    suit = self.TWfusion(emb_concat)

    return suit