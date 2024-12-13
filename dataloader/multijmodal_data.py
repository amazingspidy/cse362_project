import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import clip
import random

from torchvision import transforms
class MultiModalData(Dataset):
    def __init__(self, data_df, category_df, image_dir, question=None, mode='train'):
        self.mode = mode
        self.data_df = data_df[data_df['type'] == mode].reset_index(drop=True)
        self.category_dict = category_df.set_index('ID')['Category'].to_dict()
        self.image_dir = image_dir
        self.question = question
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.preprocess = preprocess
        self.model = model
        self.set_id_search_dict = {row['set_id']: idx for idx, row in self.data_df.iterrows()}

        self.texts = 0
        self.images = 0
    
    def preprocess_images(self, image_list): # for clip preprocess
        processed_images = [self.preprocess(image).unsqueeze(0) for image in image_list]
        return torch.cat(processed_images, dim=0)
    
    def pick_random_one(self):
        rand_idx = random.randint(0, len(self.data_df)-1)
        data_row = self.data_df.iloc[rand_idx]
        set_id = data_row['set_id']
        items = data_row['items']
        length = len(items)

        rand_idx_pick = random.randint(0, length-1)
        item = items[rand_idx_pick]
        category_id = str(item['categoryid'])
        category_name = self.category_dict.get(category_id, 'Unknown Category')
        item_name = item['name']
        text = f"{category_name}: {item_name}."

        image_folder=os.path.join(self.image_dir, str(set_id))
        image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
        image_filename = image_filenames[rand_idx_pick]
        image_path = os.path.join(image_folder, image_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {image_path}, Error: {e}")

        text_input = clip.tokenize([text]).to(self.device)
        text_output = self.model.encode_text(text_input)
        image_tensor = self.preprocess_images([image]).to(self.device)
        image_output = self.model.encode_image(image_tensor)

        return text_output, image_output
    
    def pick_random_multi(self, num):
        rand_idx = [random.randint(0, len(self.data_df)-1) for _ in range(num)]

        texts = []
        images=[]

        for idx in rand_idx:
            data_row = self.data_df.iloc[idx]
            set_id = data_row['set_id']
            items = data_row['items']
            length = len(items)

            rand_idx_pick = random.randint(0, length-1)
            item = items[rand_idx_pick]
            category_id = str(item['categoryid'])
            category_name = self.category_dict.get(category_id, 'Unknown Category')
            item_name = item['name']
            text = f"{category_name}: {item_name}."
            texts.append(text)

            image_folder=os.path.join(self.image_dir, str(set_id))
            image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
            image_filename = image_filenames[rand_idx_pick]
            image_path = os.path.join(image_folder, image_filename)
            try:
                image = Image.open(image_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Error loading image: {image_path}, Error: {e}")

        text_input = clip.tokenize(texts).to(self.device)
        text_output = self.model.encode_text(text_input)
        image_tensor = self.preprocess_images(images).to(self.device)
        image_output = self.model.encode_image(image_tensor)

        return text_output, image_output
    
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        data_row = self.data_df.iloc[idx]
        set_id = data_row['set_id']
        text_list = []
        price_list = []
        likes_list = []
        
        items = data_row['items']
        for item in items:
            category_id = str(item['categoryid'])
            category_name = self.category_dict.get(category_id, 'Unknown Category')
            item_name = item['name']
            text = f"{category_name}: {item_name}."
            text_list.append(text) 
            # price_list.append(item['price'])
            # likes_list.append(item['likes'])

        images = []
        image_folder = os.path.join(self.image_dir, str(set_id))
        if os.path.isdir(image_folder):
            image_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
            for image_filename in image_filenames[1:]: # 1.jpg부터
                image_path = os.path.join(image_folder, image_filename)
                try:
                    image = Image.open(image_path).convert('RGB')
                    images.append(image)  
                except Exception as e:
                    print(f"Error loading image: {image_path}, Error: {e}")
        else:
            print(f"Image folder not found: {image_folder}")


        max_length = min(len(text_list), len(images))  
        text_list = text_list[:max_length]  
        images = images[:max_length]

        rand_idx = [i for i in range(max_length)]
        random.shuffle(rand_idx)

        text_input = clip.tokenize(text_list).to(self.device)
        texts = self.model.encode_text(text_input)
        texts = texts[rand_idx,:].float()
        image_tensor = self.preprocess_images(images).to(self.device)
        images = self.model.encode_image(image_tensor)
        images = images[rand_idx, :].float()

        
        # 길이 부족시 padding 채움
        # 일단 패딩이 추가된다고 하면, max_length idx부터 추가가 됨.
        if max_length < 8:
            padding = torch.zeros(8 - max_length, 512).to(self.device)
            texts = torch.cat((texts, padding), dim=0)
            images = torch.cat((images, padding), dim=0)
            price_list += [-1] * (8 - max_length)
            likes_list += [-1] * (8 - max_length)
        
        
        L, D = texts.shape

        assert(L == 8)
        
        # question = []
        # gt_idx = random.randint(0, max_length-1)
        
        suit = random.randint(0,1) # 50% 확률로 치환

        # 랜덤한 text, image를 뽑고 gt_idx에 치환시켜야함 gt는 적합도이니 치환했다면 0 아니면 1로 함

        if self.mode == 'train':
            if suit == 0:
                gt_idx = [idx for idx in range(max_length)]
                random.shuffle(gt_idx)
                replace_num = random.randint(1,3)
                text_f, image_f = self.pick_random_multi(replace_num)
                texts[gt_idx[:replace_num], :] = text_f.float()
                images[gt_idx[:replace_num], :] = image_f.float()
                # texts_sliced = torch.cat((texts[ : gt_idx, :] , text_f, texts[ gt_idx+1:, :]), dim=0)
                # images_sliced= torch.cat((images[ : gt_idx, :] , image_f, images[ gt_idx+1:, :]), dim=0)
                # text_gt = texts[gt_idx, :]
                # image_gt = images[gt_idx, :]
            # else:
            #     # texts_sliced = torch.cat((texts[ : gt_idx, :] , texts[ gt_idx+1:, :]), dim=0)
            #     # images_sliced = torch.cat((images[ : gt_idx, :] , images[ gt_idx+1:, :]), dim=0)
            #     texts_sliced = texts
            #     images_sliced = images
            #     text_gt = texts[gt_idx, :]
            #     image_gt = images[gt_idx, :]
            
        # prices = torch.tensor(price_list)
        # likes = torch.tensor(likes_list)
        
        if self.mode == 'test' and self.question is not None:
            filtered_question = self.question[self.question['set_id'] == set_id]
            question_data ={}
            for _, row in filtered_question.iterrows():
                
                question_data = {
                    'question_ids': row['question'],  # 얘도 사실 필요없다.
                    'answer': row['answers'], 
                    'blank_position': row['blank_position']
                }
                #question.append(question_data)
            
            self.set_id_search_dict[set_id] = idx
            
            return {
                'texts': texts.float(),
                # 'prices': prices,
                # 'likes': likes,
                'images': images.float(), 
                'set_id': set_id, 
                'question': question_data,
                'valid_idx' : max_length-1,
                'gt_idx' : gt_idx #for gt select and train
            }
        
        return {
            'texts': texts,
            # 'text_gt': text_gt,
            # 'prices': prices,
            # 'likes': likes,
            'images': images, 
            # 'image_gt': image_gt,
            'set_id': set_id,
            'valid_idx': max_length-1,
            # 'gt_idx' : gt_idx,
            'suit' : suit
        }