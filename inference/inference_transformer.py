from model.transfusion import TransFusion
from model.transfusion import Encoder
from model.transfusion import Decoder

import torch

import pandas as pd
import os
from dataloader.sampling import DataSampler
from dataloader.multimodal_data import MultiModalData
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
path = 'data/meta/'
test = pd.read_json(path + 'test_no_dup.json')
blank = pd.read_json(path + 'fill_in_blank_test.json')
base_dir = ''
data_dir = os.path.join(base_dir, 'data')
meta_dir = os.path.join(data_dir, 'meta')
image_dir = os.path.join(data_dir, 'images')
sampler = DataSampler(data_path = meta_dir,  test_sampling_ratio=1.0)
concat_df, question_data = sampler.sample_data()
test_dataset = MultiModalData(concat_df, sampler.category_df, image_dir, question = question_data, mode='test')


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # None 값 필터링
    if len(batch) == 0:
        return None
    return {
        'images': default_collate([item['images'] for item in batch]),
        'texts': default_collate([item['texts'] for item in batch]),
        'question': default_collate([item['question'] for item in batch])
    }
    
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False)
# 1. 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"


INPUT_DIM = None
OUTPUT_DIM = None
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
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


net = model_TransFusion.to(device)




# 2. 모델 파라미터 로드
checkpoint = "transformer_epoc20_withoutnorm.pth"
net.load_state_dict(torch.load(checkpoint, map_location=device), strict=False)
net.eval()  # 모델을 추론 모드로 전환
print("Model loaded and set to eval mode.")


#blank_idx를 기반으로 하여, 기존 image embedding에서 questions를 뽑습니다.
def pick_questions(texts, images, blank_idx):
    texts_before_blank = texts[:, :blank_idx, :]
    texts_after_blank = texts[:, blank_idx+1:, :]
    images_before_blank = images[:, :blank_idx, :]
    images_after_blank = images[:, blank_idx+1:, :]
    q_texts = torch.cat((texts_before_blank, texts_after_blank), dim=1)
    q_images = torch.cat((images_before_blank, images_after_blank), dim=1)
    
    return q_texts, q_images

# answer_sheet를 입력으로 받아, ans_idx 인덱스에 해당하는 선택지의 임베딩을 뽑습니다. (선지 하나에 대한 임베딩 뽑기)
def pick_answer_embedding(out, answer_sheet, ans_idx, set_id_dict):
    value = answer_sheet[ans_idx][0]
    set_id, idx = value.split('_')
    set_id = int(set_id)
    idx = int(idx)
    data = test_dataset[set_id_dict[set_id]]
    answer_embedding = torch.cat((data['images'][idx-1, :], data['texts'][idx-1, :]),dim=0)
    return answer_embedding

# Cosine similarity 계산
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
    
# answer_sheet와 set_id_dict를 넣으면, 위에서 정의된 함수를 활용하여, 
# 모델이 출력한 out 임베딩과 cosine similarity가 가장 가까운 정답지를 선택합니다.    
def pick_proper_answer(out, answer_sheet, set_id_dict):
    similarity_score = []
    
    # out의 shape가 (1, 2, 512)이므로, 이를 (1, 1024)로 변환
    out_reshaped = out.view(1024)  
    
    for i in range(4): #선택지 4개를 순회하며 유사도 점수를 계산합니다.
        answer_embedding = pick_answer_embedding(out, answer_sheet, i, set_id_dict)
        similarity = cosine_similarity(out_reshaped, answer_embedding)
        similarity_score.append(similarity) 
    
    best_match_idx = torch.argmax(torch.stack(similarity_score))  # 가장 큰 cosine similarity를 가진 인덱스를 선택

    return best_match_idx



set_id_dict = test_dataset.set_id_search_dict # answer의 임베딩을 역으로 구해야할때 필요

correct = 0
total = 0




with torch.no_grad():  # 학습이 아니라 추론이므로 gradient 계산 비활성화
    for batch_idx, batch in enumerate(test_loader):
        # batch에서 texts와 images 가져오기
        texts = batch['texts'].to(device)
        images = batch['images'].to(device)
        blank_idx = batch['question']['blank_position'].item() - 1
        answer_sheet = batch['question']['answer']
        texts, images = pick_questions(texts, images, blank_idx)
        
        # 모델 추론
        out_images, out_texts = net.inference(model_TransFusion, images, texts, max_len=9)
        
        output = torch.cat((out_images, out_texts), dim=1)
        #print("output shape is: ",output.shape)
        # answer 뽑기
        model_answer = pick_proper_answer(output, answer_sheet, set_id_dict)

        if model_answer == 0:
            correct += 1
        total += 1
        print("model_answer is :", model_answer)
print("final score: ", correct/total * 100, "total test case: ", total)
        
       