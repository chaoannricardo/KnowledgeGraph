# -*- coding: utf8 -*-
from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy import data
from tqdm import tqdm
import codecs
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn

TRAIN_DATA_X_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\train.src"
TRAIN_DATA_y_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\train.tgt"
VALID_DATA_X_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\valid.src"
VALID_DATA_y_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\valid.tgt"
TEST_DATA_X_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\test.src"
TEST_DATA_y_PATH = "C:\\Users\\User\\Desktop\\Ricardo\\KnowledgeGraph_materials\\data_kg\\CLTS\\test.tgt"

SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


train_x_list = []
train_y_list = []
valid_x_list = []
valid_y_list = []
test_x_list = []
test_y_list = []

file_train_x = codecs.open(TRAIN_DATA_X_PATH, mode="r", encoding="utf8")
file_train_y = codecs.open(TRAIN_DATA_y_PATH, mode="r", encoding="utf8")
file_valid_x = codecs.open(VALID_DATA_X_PATH, mode="r", encoding="utf8")
file_valid_y = codecs.open(VALID_DATA_y_PATH, mode="r", encoding="utf8")
file_test_x = codecs.open(TEST_DATA_X_PATH, mode="r", encoding="utf8")
file_test_y = codecs.open(TEST_DATA_y_PATH, mode="r", encoding="utf8")


TRAIN_SAMPLE_NUM = 500
VALID_SAMPLE_NUM = 200
TEST_SAMPLE_NUM = 100

# create list for training, validation, testing set
temp_index = 0
while True:
    line = file_train_x.readline()
    train_x_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == TRAIN_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

while True or temp_index == TRAIN_SAMPLE_NUM:
    line = file_train_y.readline()
    train_y_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == TRAIN_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

while True or temp_index == VALID_SAMPLE_NUM:
    line = file_valid_x.readline()
    valid_x_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == VALID_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

while True or temp_index == VALID_SAMPLE_NUM:
    line = file_valid_y.readline()
    valid_y_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == VALID_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

while True or temp_index == TEST_SAMPLE_NUM:
    line = file_test_x.readline()
    test_x_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == TEST_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

while True:
    line = file_test_y.readline()
    test_y_list.append(line.replace(" ", "").replace("\n", ""))

    if not line or temp_index == TEST_SAMPLE_NUM:
        temp_index = 0
        break
    else:
        temp_index += 1

print(len(train_x_list), len(train_y_list), len(valid_x_list), len(valid_y_list), len(test_x_list), len(test_y_list))
print(train_x_list[10], "\n\n", train_y_list[10])


# define tokenization function
def tokenize(text):
    return [char for char in text]

SRC = Field(tokenize = tokenize,
            init_token = '<bos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

TRG = Field(tokenize = tokenize,
            init_token = '<bos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)

# building customize dataset
# reference: https://www.programmersought.com/article/7283735573/

# get_dataset constructs and returns the examples and fields required by the Dataset
def get_dataset(input_data, output_data, input_data_field, output_data_field, test=False):
	# idData pair training is useless during training, use None to specify its corresponding field
    fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("src", input_data_field), ("trg", output_data_field)]
    examples = []

    if test:
        # If it is a test set, the label is not loaded
        for text in tqdm(input_data):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(input_data, output_data)):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields

# Get the examples and fields needed to build the Dataset
train_examples, train_fields = get_dataset(train_x_list, train_y_list, SRC, TRG)
valid_examples, valid_fields = get_dataset(valid_x_list, valid_y_list, SRC, TRG)
test_examples, test_fields = get_dataset(test_x_list, test_y_list, SRC, None, test=True)

#Build Dataset dataset
train_data = data.Dataset(train_examples, train_fields)
valid_data = data.Dataset(valid_examples, valid_fields)
test_data = data.Dataset(test_examples, test_fields)

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

# activate gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

device = "cpu"

BATCH_SIZE = 1

# Use BucketIterator instead of the usual one to automatically deal with padding problem
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
                                                                      batch_size=BATCH_SIZE,
                                                                      device=device)


class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, pf_dim, dropout, learnable_pos=True, max_length=100):
        super().__init__()

        self.device = device
        # input_dim = dictionary size of the src language
        # nn.Embedding: 吃進一個 token (整數), 吐出一個 d_model 維度的向量
        print("input_dim", input_dim)
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        if learnable_pos:
            self.pos_embedding = nn.Embedding(max_length, d_model)
        else:
            self.pos_embedding = PositionalEncoding(max_length, d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, src, src_mask):
        """
        Input:
            src.shape = (batch_size, src_seq_len)
            src_mask.shape = (batch_size, 1, 1, src_seq_len)
        Output:
            src.shape = (batch_size, src_len, d_model)
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        print("src.shape", src.shape)
        print("src_mask.shape", src_mask.shape)

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # pos.shape = (batch_size, src_len)

        print("pos.shape", pos.shape)

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src.shape = (batch_size, src_len, d_model)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src


# Encoder Layer used inside Encoder module

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """
        Input:
            src.shape = (batch_size, src_len, d_model)
            src_mask.shape = (batch_size, 1, 1, src_seq_len)
        Output:
            src.shape = (batch_size, src_len, d_model)
        """
        # self-attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        return src


class PositionalEncoding(nn.Module):
    def __init__(self, max_length, d_model):
        super().__init__()

        self.d_model = d_model
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        # position.shape = (max_length, 1)

        print("position.shape", position.shape)

        angle_rates = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # angle_rates.shape = (d_model/2,)

        print("angle_rates.shape", angle_rates.shape)

        angle_rads = position * angle_rates
        # angle_rads.shape = (max_length, d_model/2)

        print("angle_rads.shape:", angle_rads.shape)

        pe[:, 0::2] = torch.sin(angle_rads)  # 取偶數
        pe[:, 1::2] = torch.cos(angle_rads)  # 取奇數
        pe = pe.unsqueeze(0)
        # pe.shape = (1, max_length, d_model)

        print("pe.shape", pe.shape)

        self.register_buffer('pe', pe)  # register a constant tensor (not updated by optim)

    def forward(self, x):
        """
        Input:
            x.shape = (batch_size, src_len)
        Output:
            x.shape = (batch_size, src_len, d_model)
        """
        x = torch.zeros(x.size(0), x.size(1), self.d_model) + self.pe[:, :x.size(1)]

        return x


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads

        # 確保d_model可以被num_heads整除
        assert d_model % self.n_heads == 0
        self.head_dim = d_model // n_heads  # 將 d_model dimension 分成 n_heads 份

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        """
        Input:
            query.shape = (batch_size, query_len, d_model)
            key.shape = (batch_size, key_len, d_model)
            value.shape = (batch_size, value_len, d_model)
            mask.shape = (batch_size, 1, query_len, key_len)
        Output:
            output.shape = (batch_size, query_len, d_model)
            attention_weights.shape = (batch_size, n_heads, query_len, key_len)
        """
        batch_size = query.shape[0]

        # 通過全連結層形成 Q,K,V
        Q = self.wq(query)
        K = self.wk(key)
        V = self.wv(value)
        # Q.shape = (batch_size, query_len, d_model)
        # K.shape = (batch_size, key_len, d_model)
        # V.shape = (batch_size, value_len, d_model)

        # 將 q,k,v 等份切成 num_heads 份
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q.shape = (batch_size, n_heads, query_len, head_dim)
        # K.shape = (batch_size, n_heads, key_len, head_dim)
        # V.shape = (batch_size, n_heads, value_len, head_dim)

        # 每個 heads 分別做 Q,K 內積
        scaled_attention_logits = torch.einsum("ijkl,ijml->ijkm", [Q, K]) / self.scale
        # scaled_attention_logits.shape = (batch_size, n_heads, query_len, key_len)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e10)

        # 得到每個 heads 的 self-attention matrix
        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        # attention_weights.shape = (batch_size, n_heads, query_len, key_len)

        output = torch.matmul(self.dropout(attention_weights), V)
        # output.shape = (batch_size, n_heads, query_len, head_dim)

        output = output.permute(0, 2, 1, 3).contiguous()
        # output.shape = (batch_size, query_len, n_heads, head_dim)

        # concat 所有 heads
        output = output.view(batch_size, -1, self.d_model)
        # output.shape = (batch_size, query_len, d_model)

        output = self.fc_out(output)
        # output.shape = (batch_size, query_len, d_model)

        return output, attention_weights


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(d_model, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input:
            x.shape = (batch_size, seq_len, d_model)
        Output:
            x.shape = (batch_size, seq_len, d_model)
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, pf_dim, dropout, max_length=100):
        super().__init__()
        # output_dim = dictionary size of the trg language
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Input:
            trg.shape = (batch_size, trg_len)
            enc_src.shape = (batch_size, src_len, d_model)
            trg_mask.shape = (batch_size, 1, trg_len, trg_len)
            src_mask.shape = (batch_size, 1, 1, src_len)
        Output:
            output.shape = (batch_size, trg_len, output_dim)
            attention_weights.shape = (batch_size, n_heads, trg_len, src_len)
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        # pos.shape = (batch_size, trg_len)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        # trg.shape = (batch_size, trg_len, d_model)

        for layer in self.layers:
            trg, attention_weights = layer(trg, enc_src, trg_mask, src_mask)
            # trg.shape = (batch_size, trg_len, d_model)
            # attention_weights.shape = (batch_size, n_heads, trg_len, src_len)

        output = self.fc_out(trg)
        # output.shape = (batch_size, trg_len, output_dim)

        return output, attention_weights


# decoder layer for decoder usage

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, pf_dim, dropout):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(d_model, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(d_model, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Input:
            trg.shape = (batch_size, trg_len, d_model)
            enc_src.shape = (batch_size, src_len, d_model)
            trg_mask.shape = (batch_size, 1, trg_len, trg_len)
            src_mask.shape = (batch_size, 1, 1, src_len)
        Output:
            trg.shape = (batch_size, trg_len, d_model)
            attention_weights.shape = (batch_size, n_heads, trg_len, src_len)
        """
        # self-attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, d_model)

        # encoder attention
        _trg, attention_weights = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # attention_weights.shape = (batch_size, n_heads, trg_len, src_len)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, d_model)

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg.shape = (batch_size, trg_len, d_model)

        return trg, attention_weights


class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src.shape = (batch_size, src_len)

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask.shape = (batch_size, 1, 1, src_len)

        return src_mask

    def make_trg_mask(self, trg):
        # trg.shape = (batch_size, trg_len)

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask.shape = (batch_size, 1, 1, trg_len)

        trg_len = trg.shape[1]
        # # 製造上三角矩陣
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        # trg_sub_mask.shape = (trg_len, trg_len)

        trg_mask = trg_pad_mask & trg_sub_mask  # AND operation: T & T = T, T & F = F, F & F = F
        # trg_mask.shape = (batch_size, 1, trg_len, trg_len)

        return trg_mask

    def forward(self, src, trg):
        """
        Input:
            src.shape = (batch_size, src_len)
            trg.shape = (batch_size, trg_len)
        Output:
            output.shape = (batch_size, trg_len, output_dim)
            attention_weights.shape = (batch_size, n_heads, trg_len, src_len)
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask.shape = (batch_size, 1, 1, src_len)
        # trg_mask.shape = (batch size, 1, trg_len, trg_len)

        enc_src = self.encoder(src, src_mask)
        # enc_src.shape = (batch_size, src_len, d_model)

        output, attention_weights = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output.shape = (batch_size, trg_len, output_dim)
        # attention_weights.shape = (batch_size, n_heads, trg_len, src_len)

        return output, attention_weights


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
D_MODEL = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
N_EPOCHS = 15
CLIP = 1
# MAX_LENGTH_INPUT = int(np.max([len(data)+2 for data in train_x_list]))
# MAX_LENGTH_OUTPUT = int(np.max([len(data)+2 for data in train_y_list]))
MAX_LENGTH_INPUT = 10
MAX_LENGTH_OUTPUT = 10
LEARNING_RATE = 0.001

# 建立 encoder 和 decoder class
enc = Encoder(INPUT_DIM, D_MODEL, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, MAX_LENGTH_INPUT)
dec = Decoder(OUTPUT_DIM, D_MODEL, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, MAX_LENGTH_OUTPUT)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]  # PAD_IDX=1

model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX).to(device)


# Model summary
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# print(model.apply(init_weights))
print(f'The model has {count_parameters(model):,} trainable parameters')


class ScheduledOptim():
    '''
    A wrapper class for optimizer
    From https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
    '''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-09),
                           d_model=D_MODEL, n_warmup_steps=4000)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train_epoch(model, iterator, optimizer, criterion, clip):
    # train mode
    model.train()

    epoch_loss = 0
    for i, batch in enumerate(iterator):
        src = batch.src
        # src.shape = (batch_size, src_len)

        trg = batch.trg
        # trg.shape = (batch_size, trg_len)

        # 梯度歸零
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])  # full teacher forcing
        # output.shape = (batch_size, trg_len-1, output_dim)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        # output.shape = ((trg_len-1) * batch_ize, output_dim)
        # trg.shape = ((trg_len-1) * batch_size)

        # 計算 loss
        loss = criterion(output, trg)  # outputs by default are from logits; trg no need to do one-hot encoding
        # 反向傳播，計算梯度
        loss.backward()
        # 做 regularization，使得整體梯度 norm 不超過 1，以防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # 更新優化器
        optimizer.step_and_update_lr()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_epoch(model, iterator, criterion):
    # evaluation mode
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            # src.shape = (src_len, batch_size)

            trg = batch.trg
            # trg.shape = (trg_len, batch_size)

            output, _ = model(src, trg[:, :-1])  # turn off teacher forcing
            # output.shape = (trg_len, batch_size, output_dim)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            # output.shape = ((trg_len-1) * batch_ize, output_dim)
            # trg.shape = ((trg_len-1) * batch_size)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 計算跑一個 Epoch 的時間
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


# 將 loss vs. Epoch 畫出來
def showPlot(tr_points, va_points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(tr_points, label='train loss')
    plt.plot(va_points, label='validation loss')
    plt.legend()
    plt.xlabel('epoch')


def train(model, train_iterator, valid_iterator, optimizer, criterion):
    best_valid_loss = float('inf')
    plot_tr_loss = []
    plot_va_loss = []
    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate_epoch(model, valid_iterator, criterion)
        plot_tr_loss.append(train_loss)
        plot_va_loss.append(valid_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # 儲存模型 (只存權重)
            torch.save(model.state_dict(), 'SavedModel/tr-model_0720.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # PPL 是 perplexity 的縮寫，基本上就是 cross-entropy 指數化；其值越小越好 (minimize probability likelyhood)
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    showPlot(plot_tr_loss, plot_va_loss)


train(model, train_iterator, valid_iterator, optimizer, criterion)