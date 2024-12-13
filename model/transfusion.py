import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DBG = False

class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        ''' Input Embedding '''
        # self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        ''' Multiple Encoder Layers '''
        # we use multiple encoder layers (e.g., 6 in the original Transformer paper)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, input_src, input_mask):

        batch_size = input_src.shape[0]
        src_len = input_src.shape[1]
        # emb_output = self.tok_embedding(input_src)
        pos_tensor = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # 스케일 값 사용 확인 필요
        emb_output = input_src.to(self.device)
        output = self.dropout(emb_output * self.scale + self.pos_embedding(pos_tensor))
        for layer in self.layers:
          output = layer(output, input_mask)

        return output
    

class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        ''' Multi Head self-Attention '''
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Positional FeedForward Layer'''
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        # src : (batch_size, src_length, hidden_dim)
        # _src : (batch_size, src_length, hidden_dim)
        _src, attention = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(self.dropout(_src) + src)
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(self.dropout(_src) + src)


        return src
    
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fcQ = nn.Linear(hid_dim, hid_dim)
        self.fcK = nn.Linear(hid_dim, hid_dim)
        self.fcV = nn.Linear(hid_dim, hid_dim)
        self.fcOut = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]
        src_len = query.shape[1]
        k_len = key.shape[1]
        v_len = value.shape[1]
        # (b, s_l, n_h, h_d)
        Q = self.fcQ(query).view(batch_size, src_len, self.n_heads, self.head_dim)
        K = self.fcK(key).view(batch_size, k_len, self.n_heads, self.head_dim)
        V = self.fcV(value).view(batch_size, v_len, self.n_heads, self.head_dim)

        # (b, n_h, s_l, h_d)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)

        scaled_atten = (Q / (self.head_dim ** 0.5)) @ K.transpose(2, 3) # (b, n_h, s_l, s_l)
        if DBG: print("mask size:",mask.shape)
        if DBG: print("sc_attn:",scaled_atten.shape)
        if mask is not None:
            scaled_atten = scaled_atten.masked_fill(mask == 0, -1e9)
        attention = self.dropout(nn.functional.softmax(scaled_atten, dim=-1))
        # attention = nn.functional.softmax(scaled_atten, dim=-1)
        # print("V:",V.shape)
        # print("attn:",attention.shape)
        output = torch.matmul(attention, V)
        x = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fcOut(x)
        # output = self.dropout()
        x = self.dropout(x)




        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)


    def forward(self, x):

        x = self.fc1(x)
        x = self.dropout(torch.relu(x))
        x = self.fc2(x)


        return x

class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device

        # self.tok_embedding = nn.Embedding(output_dim, hid_dim)

        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        # self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)




    def forward(self, trg_seq, trg_mask, enc_output, src_mask):

        batch_size = trg_seq.shape[0]
        seq_len = trg_seq.shape[1]
        # emb_output = self.tok_embedding(trg_seq)
        emb_output = trg_seq
        pos_tensor = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        emb_output = emb_output.to(self.device)
        output = self.dropout(emb_output * self.scale + self.pos_embedding(pos_tensor))
        for layer in self.layers:
          output, attention = layer(output, enc_output, trg_mask, src_mask)
        # output = self.fc_out(output)



        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        ''' Multi Head self Attention'''
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Encoder-decoder attention'''
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)

        ''' Positionwise FeedForward Layer'''
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

      _output, atten = self.self_attention(trg, trg, trg, trg_mask)
      output = self.self_attn_layer_norm(self.dropout(_output) + trg)
      _output, enc_dec_atten = self.encoder_attention(output, enc_src, enc_src, src_mask)
      output = self.enc_attn_layer_norm(self.dropout(_output) + output)
      _output = self.positionwise_feedforward(output)
      #trg = self.ff_layer_norm(self.dropout(_output) + output)
      trg = self.dropout(_output) + output

      attention = enc_dec_atten


      return trg, attention
    

class TransFusion(nn.Module):
    def __init__(self,
                 encoder,
                 decoder_img,
                 decoder_txt,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder_img = decoder_img
        self.decoder_txt = decoder_txt
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.fc1_out = nn.Linear(512*7, 512)
        self.fc2_out = nn.Linear(512*7, 512)
        self.device = device
    
    def inference(self, model, images, texts, max_len=9):
        model.eval()
        batch_size, trg_len, _ = images.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, trg_len, trg_len), device=device), diagonal=1)).bool()
        src = torch.cat((images, texts), dim=1)
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        src_mask = src_mask.any(dim=-1)
        # trg_indexes = [-2.0]

        trg_tensor_img = torch.zeros(batch_size, 1, 512).to(device)
        trg_tensor_txt = torch.zeros(batch_size, 1, 512).to(device)
        enc_out = model.encoder(src, src_mask)
        for i in range(max_len):
            # trg_tensor = torch.FloatTensor(trg_indexes).unsqueeze(0).to(device)
            # print(trg_tensor.shape)
            trg_len = trg_tensor_img.shape[1]
            trg_temp = trg_tensor_img[..., 0]
            subsequent_mask = (1 - torch.triu(torch.ones((1, trg_len, trg_len), device=device), diagonal=1)).bool()
            trg_mask = (trg_temp != self.trg_pad_idx).unsqueeze(1) & subsequent_mask
            output_img, attention_img = model.decoder_img(trg_tensor_img, trg_mask, enc_out, src_mask)
            output_txt, attention_txt = model.decoder_txt(trg_tensor_txt, trg_mask, enc_out, src_mask)
            if DBG: print("output_img:",output_img.shape, "output_txt:",output_txt.shape)
            trg_tensor_img = torch.cat((trg_tensor_img, output_img[:,-1,:].unsqueeze(1)), dim=1)
            trg_tensor_txt = torch.cat((trg_tensor_txt, output_txt[:,-1,:].unsqueeze(1)), dim=1)
        pred_img = output_img[:,-1,:].unsqueeze(1)
        pred_txt = output_txt[:,-1,:].unsqueeze(1)
        return pred_img, pred_txt


    def forward(self, src, src2, trg, trg2):
        
        batch_size, trg_len, _ = trg.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, trg_len, trg_len), device=self.device), diagonal=1)).bool()
        src = torch.cat((src, src2), dim=1)
        # print("src:",src.shape)
        src_mask = (src != self.src_pad_idx).unsqueeze(1)
        # print("s_mask:",src_mask.shape)
        src_mask = src_mask.any(dim=-1)
        # print("s_mask:",src_mask.shape)
        trg_temp = trg[..., 0]
        # print(trg_temp.shape)
        trg_mask = (trg_temp != self.trg_pad_idx).unsqueeze(1) & subsequent_mask
        if DBG: print("src:",src.shape)
        if DBG: print("s_mask:",src_mask.shape)
        # if DBG: print("t_mask:",trg_mask.shape)
        

        enc_out = self.encoder(src, src_mask)
        # print("enc_out:",enc_out.shape)
        # return enc_out
        output_image, attention_image = self.decoder_img(trg, trg_mask, enc_out, src_mask)
        output_text, attention_text = self.decoder_txt(trg2, trg_mask, enc_out, src_mask)
        return output_image, output_text