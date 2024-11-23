from torch import nn
import torch

class Multifusion(nn.Module):
  def __init__(self):
    super(Multifusion, self).__init__()
    ##### TODO #####
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.encode_meta = nn.Linear(2, 512, bias=True)
    self.ffn = nn.Linear(512 * 3 , 512)
    
  def forward(self, image_embedding, text_embedding, prices, likes):
    ##### TODO #####

    batch_size = image_embedding.shape[0]      
    meta_feature = torch.stack([prices, likes], dim=-1) # [B, 8, 2]
    meta_embedding = self.encode_meta(meta_feature) # [B, 8, 512]
    
    # 특징들을 concatenate
    fused_feature = torch.cat((image_embedding, text_embedding, meta_embedding), dim=1)  # [B, 8, 3, 512]
    fused_vec   = fused_feature.view(batch_size, 8,  -1)    # [B, 8, 512 * 3]
    
    # FFN으로 임베딩크기 축소
    out = self.ffn(fused_vec)  # [batch_size, 8, 512]
    return out