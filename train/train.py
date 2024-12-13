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
from model.multifusion import Multifusion
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None 값 필터링
    if len(batch) == 0:
        return None
    return {
        'images': default_collate([item['images'] for item in batch]),
        'texts': default_collate([item['texts'] for item in batch]),
        'image_gt': default_collate([item['image_gt'] for item in batch]),
        'text_gt': default_collate([item['text_gt'] for item in batch]),
    }
    
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
valid_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, mode='valid')
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = config['training']['batch_size']
lr = config['training']['learning_rate']


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False)

model = Multifusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

def cosine_similarity_loss(image_context, image_gt, text_context, text_gt):
    # 이미지 임베딩의 Cosine Similarity 계산 및 손실 정의
    image_similarity = F.cosine_similarity(image_context, image_gt, dim=-1)
    loss_image = 1 - image_similarity.mean()  # 유사도를 최대화
    
    # 텍스트 임베딩의 Cosine Similarity 계산 및 손실 정의
    text_similarity = F.cosine_similarity(text_context, text_gt, dim=-1)
    loss_text = 1 - text_similarity.mean()  # 유사도를 최대화

    # 최종 손실 합산
    total_loss = loss_image + loss_text
    return total_loss

## Validation 코드
def validate(model, dataloader, device):
    model.eval()  # 모델을 평가 모드로 설정
    total_loss = 0
    total_batches = 0

    with torch.no_grad():  # Gradient 계산 비활성화
        for batch_idx, batch in enumerate(dataloader):
            # 데이터 로드
            image_embedding = batch['images'].to(device)  # Shape: (B, 7, 512)
            text_embedding = batch['texts'].to(device)    # Shape: (B, 7, 512)
            image_gt = batch['image_gt'].to(device)       # Shape: (B, 1, 512)
            text_gt = batch['text_gt'].to(device)         # Shape: (B, 1, 512)

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
        image_embedding = batch['images'].to(device) # Shape: (B, 7, 512)
        text_embedding = batch['texts'].to(device)  # Shape: (B, 7, 512)
        image_gt = batch['image_gt'].to(device)   # Shape: (B, 1, 512)
        text_gt = batch['text_gt'].to(device)      # Shape: (B, 1, 512)
        
        # Forward pass
        output = model(image_embedding, text_embedding)  # Shape: (B, 2, 512)

        # Output 분리
        image_context = output[:, 0, :]  # Shape: (B, 512)
        text_context = output[:, 1, :]   # Shape: (B, 512)

        # 손실 계산
        loss = cosine_similarity_loss(image_context, image_gt, text_context, text_gt)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 로그 출력
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Validation Step:")
    validate(model, valid_dataloader, device)

save_dir = config['model']['save_dir']   
torch.save(model.state_dict(), save_dir)
print("Model saved to multifusion_model.pth")