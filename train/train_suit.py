import pandas as pd
import torch
import clip
from PIL import Image
from importlib import reload
from dataloader.sampling import DataSampler  # 이미 import된 모듈
from importlib import reload
from dataloader.multimodal_data_suit_random_multi_mask import MultiModalData
import os
from torch.utils.data import Dataset, DataLoader
# from model.multifusion import Multifusion
# from model.multifusion_front_token import Multifusion
from model.multifusion_suit import Multifusion

path = 'data/meta/'
train = pd.read_json(path + 'train_no_dup.json')
valid = pd.read_json(path + 'valid_no_dup.json')
test = pd.read_json(path + 'test_no_dup.json')
blank = pd.read_json(path + 'fill_in_blank_test.json')
device = 'cuda'


base_dir = ''
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir,  test_sampling_ratio=0.33)
concat_df, question_data = sampler.sample_data()



train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')


g = "cuda" if torch.cuda.is_available() else "cpu"

dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
net = Multifusion().to(device)
# net.load_state_dict(torch.load('checkpoints_suit/epoch55.pth'))
criterion = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), 0.0005)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (0.95) ** epoch)

import os
os.makedirs('./checkpoints_suit_mlp',exist_ok=True)
for epoch in range(200):
    net.train()
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        image_embedding = batch['images'] # Shape: (B, 7, 512)
        text_embedding = batch['texts']  # Shape: (B, 7, 512)
        # image_gt = batch['image_gt']    # Shape: (B, 1, 512)
        # text_gt = batch['text_gt']      # Shape: (B, 1, 512)
        batch_size = image_embedding.shape[0]

        context_vector = net(image_embedding, text_embedding)
        context_vectror = context_vector.squeeze(-1)

        # gt = batch['suit'].to(torch.float32).to(device)
        # loss1 = criterion(context_vector[:,0].view(batch_size, 1), gt.view(batch_size, 1))
        # gtt = batch['gt_idx'].to(device)
        # mask = torch.nonzero(1-gt)[:,0]
        # loss2 = criterion2(context_vector[mask,1:], gtt[mask])
        
        # loss = 0.9 * loss1 + 0.1 * loss2
        gt = batch['suit'].to(torch.float32).to(device)
        loss = criterion(context_vector[:,0].view(batch_size, 1), gt.view(batch_size, 1))
        loss.backward()
        optimizer.step()

        print(f'\rEpoch {epoch} Loss: {loss:.12f}', end='')# Suit-Loss: {loss1:.4f} Which-Loss:{loss2:.4f}', end='')

        # print("images: " , batch['images'].shape)
        # print("texts:" , batch['texts'].shape)
        # print("prices: ", batch['prices'].shape)
        # print("likes: ", batch['likes'].shape)
        # print("valid_idx: ", batch['valid_idx'].shape)
        # print("gt_idx: ", batch['gt_idx'].shape)
        # fused_output = net(batch['images'], batch['texts'], batch['prices'], batch['likes'])
        # print("fused_output: ", fused_output.shape)
    
    scheduler.step()

    net.eval()
    loss = 0
    total=0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            image_embedding = batch['images'] # Shape: (B, 7, 512)
            text_embedding = batch['texts']  # Shape: (B, 7, 512)
            # image_gt = batch['image_gt']    # Shape: (B, 1, 512)
            # text_gt = batch['text_gt']      # Shape: (B, 1, 512)
            batch_size = image_embedding.shape[0]

            context_vector = net(image_embedding, text_embedding)
            context_vectror = context_vector.squeeze(-1)

            gt = batch['suit'].to(torch.float32).to(device)
            loss += criterion(context_vector[:,0].view(batch_size, 1), gt.view(batch_size, 1))
            # gtt = batch['gt_idx'].to(device)
            # mask = torch.nonzero(1-gt)[:,0]
            # loss2 = criterion2(context_vector[mask,1:], gtt[mask])
            # loss += loss1
            total+=1

            
    print(f' Valid-Loss: {loss/total:.12f}', end='')
    # for batch_idx, batch in enumerate(validloader):
    #     pass
    print()
    torch.save(net.state_dict(), f'./checkpoints_suit_mlp/epoch{epoch}.pth')
    
# torch.save(net.state_dict(), f'./checkpoints_suit/epoch{epoch}.pth')
print("finish")

