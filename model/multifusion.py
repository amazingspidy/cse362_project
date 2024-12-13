from torch import nn
import torch

class Multifusion(nn.Module):
  def __init__(self):
    super(Multifusion, self).__init__()
    ##### TODO #####
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    self.ffn_image = nn.Linear(512 * 7, 512)
    self.ffn_text = nn.Linear(512 * 7, 512)
    
  def forward(self, image_embedding, text_embedding):
    ##### TODO #####

    batch = image_embedding.shape[0]
    
    # image_embedding = torch.cat((image_embedding[:, : gt_idx, :] , image_embedding[:, gt_idx+1:, :]), dim=1)
    # text_embedding = torch.cat((text_embedding[:, : gt_idx, :] , text_embedding[:, gt_idx+1:, :]), dim=1)
    
    image_embedding = torch.flatten(image_embedding, 1)
    text_embedding = torch.flatten(text_embedding, 1)
    
    # 특징들을 concatenate
    image_context = self.ffn_image(image_embedding).unsqueeze(1) #[B, 1, 512]
    text_context = self.ffn_text(text_embedding).unsqueeze(1)    #[B, 1, 512]
    
    out = torch.cat((image_context, text_context), dim=1) #[B, 2, 512]

    return out