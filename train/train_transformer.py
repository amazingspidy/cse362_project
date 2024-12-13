import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import clip
from PIL import Image
from importlib import reload
from dataloader.sampling import DataSampler  # 이미 import된 모듈
from importlib import reload
from dataloader.multimodal_data import MultiModalData
import os
from torch.utils.data import Dataset, DataLoader
from model.transfusion import TransFusion
from model.transfusion import Encoder
from model.transfusion import Decoder
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config = load_config("config.yaml")
train = config['dataset']['train']
valid = config['dataset']['valid']
test = config['dataset']['test']
blank = config['dataset']['blank']
data_dir = config['dataset']['data_dir']
meta_dir = config['dataset']['meta_dir']
image_dir = config['dataset']['image_dir']
train_sampling_ratio = config['dataset']['train_sampling_ratio']
valid_sampling_ratio = config['dataset']['valid_sampling_ratio']
test_sampling_ratio = config['dataset']['test_sampling_ratio']
sampler = DataSampler(data_path = meta_dir,  train_sampling_ratio=train_sampling_ratio,valid_sampling_ratio=valid_sampling_ratio,test_sampling_ratio=test_sampling_ratio)
concat_df, question_data = sampler.sample_data()


train_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='train')
print("train dataset len: ", len(train_dataset))
valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = config['training']['batch_size']
lr = config['training']['learning_rate']


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)



INPUT_DIM = None
OUTPUT_DIM = None
HID_DIM = 512
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
enc = Encoder(INPUT_DIM,
            HID_DIM,
            ENC_LAYERS,
            ENC_HEADS,
            ENC_PF_DIM,
            ENC_DROPOUT,
            device)
dec_img = Decoder(OUTPUT_DIM,
          HID_DIM,
          DEC_LAYERS,
          DEC_HEADS,
          DEC_PF_DIM,
          DEC_DROPOUT,
          device)
dec_txt = Decoder(OUTPUT_DIM,
          HID_DIM,
          DEC_LAYERS,
          DEC_HEADS,
          DEC_PF_DIM,
          DEC_DROPOUT,
          device)
SRC_PAD_IDX = -1
TRG_PAD_IDX = -1
model_TransFusion = TransFusion(enc, dec_img, dec_txt, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model_TransFusion.parameters(), lr=lr)



## Validation 코드
def validate(model, dataloader, device):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    total_batches = 0

    with torch.no_grad():  # Gradient 계산 비활성화
        for batch_idx, batch in enumerate(dataloader):
            # 데이터 로드
            image_embedding = batch['images'].to(device)  # Shape: (B, 8, 512)
            text_embedding = batch['texts'].to(device)    # Shape: (B, 8, 512)
            print("image_emb size: ", image_embedding.shape)
            # Forward pass
            output = model(image_embedding, text_embedding)  # Shape: (B, 2, 512)

            # Output 분리
            image_context = output[:, 0, :]  # Shape: (B, 512)
            text_context = output[:, 1, :]  # Shape: (B, 512)

            # 손실 계산
            loss = cosine_similarity_loss(image_context, image_gt, text_context, text_gt)

            # 손실 축적
            total_loss += loss.item()
            total_batches += 1

    # 평균 손실 계산
    avg_loss = total_loss / total_batches
    print(f"Validation Loss: {avg_loss:.4f}")

    return avg_loss

num_epochs = config['training']['epochs']

# 학습 루프
for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_dataloader):
        # 데이터 로드
        image_embedding = batch['images_full'].to(device) # Shape: (B, 8, 512)
        text_embedding = batch['texts_full'].to(device)  # Shape: (B, 8, 512)
        image_gt = batch['image_gt'].to(device)   # Shape: (B, 1, 512)
        text_gt = batch['text_gt'].to(device)      # Shape: (B, 1, 512)
        
        padding = torch.zeros(batch_size, 1, 512).to(device)
        image_trg = torch.cat((padding, image_embedding), dim=1)
        text_trg = torch.cat((padding, text_embedding), dim=1)
        # Forward pass
        image_out, text_out  = model_TransFusion(image_embedding, text_embedding, image_trg, text_trg)  # Shape: (B, 2, 512)

        #print(image_out.shape, text_out.shape)
        # 아웃풋과 타겟과의 점수 계산 예시
        score_image = torch.matmul(image_out, image_trg.transpose(1, 2))
        score_text = torch.matmul(text_out, text_trg.transpose(1, 2))
        #print(score_image.shape, score_text.shape)
        size = score_image.shape[1]
        label = torch.arange(size).unsqueeze(0).expand(batch_size, size).to(device)
        # 로스 예시
        loss_image = criterion(score_image.reshape(-1, size), label.reshape(-1))
        loss_text = criterion(score_text.reshape(-1, size), label.reshape(-1))
        #print(loss_image, loss_text)
        #print(label)

        total_loss = loss_image + loss_text
        # 역전파 및 최적화
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 로그 출력
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {total_loss.item():.4f}")
    # print(f"Epoch [{epoch+1}/{num_epochs}] Validation Step:")
    # validate(model, valid_dataloader, device)

save_dir = config['model']['save_dir']   
torch.save(model_TransFusion.state_dict(), save_dir)
print("Model saved to multifusion_model.pth")