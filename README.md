# cse362_project
2024 Fall CSE362 team project (team6)
------
### data
<img src="https://github.com/user-attachments/assets/9e9c754c-41a8-4419-be30-4d2d3de22332" width="800" height="400"/>
  
데이터 여기서 다운받으세요!    
구글드라이브: https://drive.google.com/drive/folders/1ytLlD7xrMDYSq8r60VKi6F70mdmcIKbD![image](https://github.com/user-attachments/assets/51a4ae43-7dd1-437a-ba24-56d02b4a502f)


### dataloader/multimodal_data.py
데이터셋으로 부터 데이터를 로드합니다. 배치단위로 가져오기 때문에, 맨 앞차원은 Batch로 생각하시면 됩니다.

#### test dataset
테스트 데이터셋 즉, batch는 다음과 같습니다.   
{      
    'texts': texts,   (B, 8, 512)   
    'prices': prices,   (B, 8)   
    'likes': likes,     (B, 8)   
    'images': images,    (B, 8, 512)      
    'set_id': set_id,    (B)   
    'question': question,    
    'valid_idx' : max_length-1,   (B)   
    'gt_idx' : gt_idx    (B)    
}   

**texts, images:** 여기서 texts와 images는 data loader단에서 이미 CLIP을 적용하여 임베딩 형태로 존재합니다.   
**likes, prices:** 아직 합쳐지기 전입니다.   
**valid_idx:**  데이터셋의 패딩부분을 구별하기 위한 용도입니다. 실제로 모든 데이터쌍이 8개로 존재하지 않기때문에 8개보다 적은 데이터쌍인 경우,      
패딩을 넣어놨습니다. 따라서 모델 학습시, valid_idx를 기준으로, 어느 인덱스까지가 유효한 데이터인지 알 수 있습니다.   
**gt_idx:** gt로 가져갈 인덱스입니다.    

#### train/valid dataset
test dataset에서 question만 없다 뿐이지, 똑같습니다.



### model/multifusion.py
prices와 likes를 stack한뒤, FFN layer를 통해 512차원으로 확장시킵니다. 그러면 차원이 (B, 8, 512)가 되며, 이것이 meta_embedding입니다.   
그러면, 기존의 texts와, images는 차원이 (B, 8, 512)였기 때문에 세 임베딩의 차원이 동일해집니다.    
세 임베딩을 concat한 뒤 (B, 8, 512*3), FFN layer를 통해 (B, 8, 512)로 축소합니다.    
최종적으로 (B, 8, 512) 차원의 데이터가 나옵니다.   

이 코드를 통해, 모델에 최종적으로 들어갈 데이터의 형태가 완성됩니다.   

### test.py 
현재 test.py에서 배치단위로 데이터가 어떤 shape인지 확인할 수 있으며, 출력해보기도 가능합니다.


----
## 앞으로의 방향


### 랜덤마스킹
이것은 batch['gt_idx']의 값으로 결정되며, 데이터로드시 정해집니다.         
구체적으로는 (B, 8, 512) 차원의 데이터에서, 일단 먼저 batch['valid_idx']를 통해 데이터의 두번째차원이 어디까지가 유효한 차원인지 확인합니다.    
예를 들어 batch['valid_idx'][0] 의 값이 8이라면, 0번째 배치에서의 데이터는 착장의 개수가 8개라는 것입니다.     
8보다 적은 숫자라면, 패딩이 포함된 데이터이기 때문에, 해당(n번째) 배치에서의 착장은 batch['valid_idx'][n] 의 수만큼으로 이루어져있다고 생각하면 됩니다.    

<img src="https://github.com/user-attachments/assets/74363127-c1ab-490a-8297-8fa35eeb5259" width="800" height="500"/>

### 학습방법
기본적으로 배치단위로 학습이 이루어지기 때문에, 배치를 고려하여 차원을 적었습니다.
아래 그림에서 N이 의미하는 바는 batch['valid_idx']값에 +1을 한 값입니다. (valid_idx는 인덱스이기 때문에 1만큼 적음)    
즉 패션착장 데이터가 N개의 아이템으로 이루어져있다는 것이고, 그중 하나는 GT이기 때문에 따로 떼어집니다.    
모델은 N-1개의 데이터를 가지고 , 1개의 GT데이터를 예측하도록 학습합니다.    
<img src="https://github.com/user-attachments/assets/78b496a6-f564-4454-b193-2039136cb5f0" width="800" height="500"/>


### 추론방법
학습과 마찬가지입니다.   
이번에는 인풋으로 들어오는 N개의 데이터를 가지고, 모델이 임베딩 한개를 예측합니다.   
해당 임베딩과 가장 가까운 4지선다 임베딩중 하나를 선택하면 됩니다.    
4지선다 아이템들도, 모두 임베딩으로 전환되는 과정을 거쳐야 합니다.    
<img src="https://github.com/user-attachments/assets/23fde4a4-715d-4c28-88ae-b5a07b074271" width="800" height="500"/>

