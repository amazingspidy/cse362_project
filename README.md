
# cse362_project
2024 Fall CSE362 team project (team6)
------
### data
<img src="https://github.com/user-attachments/assets/9e9c754c-41a8-4419-be30-4d2d3de22332" width="800" height="400"/>
  
Download data!   
GoogleDrive: https://drive.google.com/drive/folders/1ytLlD7xrMDYSq8r60VKi6F70mdmcIKbD![image](https://github.com/user-attachments/assets/51a4ae43-7dd1-437a-ba24-56d02b4a502f)


### dataloader/multimodal_data.py
Load data from the dataset. It is imported in batches, so you can think of the front dimension as a Batch.


#### After data loading

Train(Valid, Test) Dataset looks like..    

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



## Overview of Our Project
<img width="857" alt="스크린샷 2024-12-12 오후 10 38 57" src="https://github.com/user-attachments/assets/ec4473ef-92fe-4501-94e2-f0aaf25679ad" />

## Overview of Inference
<img width="854" alt="스크린샷 2024-12-12 오후 10 39 09" src="https://github.com/user-attachments/assets/5f00e67e-3a2f-4126-b035-2d7182ebef1c" />
