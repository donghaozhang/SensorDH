import os
import pandas as pd
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import Dataset
import torch as tf
from collections import Counter
from datetime import datetime
import math
from sklearn.metrics import confusion_matrix
from tools import load_sensor_data_without_h, sample_sensor_data, manual_lable_array_list, most_frequent, extract_sensor_data, get_sensor_data, sample_data, sample_label, combine_to_one
from model import Classifier_dh
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, feature_len = 39, seq_len = 128, num_classes = 6, dim = 39, depth = 2, heads = 8, mlp_dim = 1024, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.5, emb_dropout = 0.5):
        super().__init__()
        #assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = seq_len
        patch_dim = feature_len
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        #self.to_patch_embedding = nn.Sequential(
           #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            #nn.Linear(patch_dim, dim),
        #)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        #x = self.to_patch_embedding(img)
        x = img
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

txt_to_label = {'talk':0, 'eat':1, 'read':2, 'drink':3, 'computer':4, 'write':5, 'other': 6}
actor_1 = [1,2,3,4,5,6,7,8,9,10]
actor_2 = [11,12,13,14,31,32,33,34,35,36]
actor_3 = [15,16,17,18,37,38,39,40,41,42]
actor_4 = [43,44,45,46,47,48,49,50,51,52]
actor_5 = [53,54,55,56,57,58,59,60,61,62]
actor_6 = [63,64,65,66,67,68,69,70,71,72]
actor_7 = [76,77,78,79,80,81,82,83,84,85]
actor_8 = [86,87,88,89,90,91,92,93,94,95]
actor_9 = [97,98,99,100,101,102,103,104,105,106]
actor_10 = [107,108,109,110,111,112,113,114,115,116]
actor_11 = [117,118,119,120,121,122,123,124,125,126]
actor_12 = [128,129,130,131,132,133,134,135,136,137]
actor_13 = [138,139,140,141,142,143,144,145,146,147]
actor_14 = [148,149,150,151,152,153,154,155,156,157]
actor_15 = [161,162,163,164,165,166,167,168,169,170]
actor_16 = [171,172,173,174,175,176,177,178,179,180]
actor_17 = [181,182,183,184,185,186,187,188,189,190]
actor_18 = [191,192,193,194,195,196,197,198,199,200]
actor_19 = [201,202,203,204,205,206,207,208,209,210]
actor_20 = [212,213,214,215,216,217,218,219,220,221]
actor_21 = [223,224,225,226,227,228,229,230,231,232]
actor_22 = [233,234,235,236,237,238,239,240,241,242]
actor_23 = [243,244,245,246,247,248,249,250,251,252]
actor_24 = [254,255,256,257,258,259,260,261,262,263]
actor_25 = [264,265,266,267,268,269,270,271,272,273]
combine_list = actor_1 + actor_2 + actor_3 + actor_4 + actor_5 + actor_6 + actor_7 + actor_8 + actor_9 + actor_10+ actor_11 + actor_12 + actor_13 + actor_14 + actor_15+ actor_16 + actor_17 + actor_18 + actor_19 + actor_20+ actor_21 + actor_22 + actor_23 + actor_24 + actor_25
window_sz = 128
sample_sz = 128
lr = 0.0001
n_epochs = 20
num_classes = 6
patience, trials = 5, 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss(reduction='sum')
bs = 256
sample_sensor_data = get_sensor_data('../save_data/sample_data/')
sample_sensor_label = get_sensor_data('../save_data/sample_label/')

print(sample_sensor_data[1].shape)
bestacc = []
for actor in range(0,25):
    total_num_list = total_num_list = actor_1 + actor_2 + actor_3 + actor_4+actor_5+actor_6+actor_7+actor_8+actor_9+actor_10+actor_11+actor_12+actor_13+actor_14+actor_15+actor_16+actor_17+actor_18+actor_19+actor_20+actor_21+actor_22+actor_23+actor_24+actor_25
    val_num_list = total_num_list[actor*10 : actor*10+10]
    del total_num_list[actor*10 : actor*10+10]
    train_num_list = total_num_list
    train_sensor_data = combine_to_one(sample_sensor_data,train_num_list)
    val_sensor_data = combine_to_one(sample_sensor_data,val_num_list)
    train_sensor_label = combine_to_one(sample_sensor_label,train_num_list)
    val_sensor_label = combine_to_one(sample_sensor_label,val_num_list)
    model = ViT().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    train_ds = TensorDataset(torch.tensor(train_sensor_data).float(), torch.tensor(train_sensor_label).long())
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_ds = TensorDataset(torch.tensor(val_sensor_data).float(), torch.tensor(val_sensor_label).long())
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=True, num_workers=0)
    best_acc = 0
    loss_history = []
    acc_history = []
    print('Start model training')
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(train_dl):
            x_raw, y_batch = [t.to(device) for t in batch]
            y_batch = tf.squeeze(y_batch)
            opt.zero_grad()
            out = model(x_raw)
            loss = criterion(out, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()
        loss_history.append(epoch_loss)
        model.eval()
        correct, total = 0, 0
        for batch in val_dl:
            x_raw, y_batch = [t.to(device) for t in batch]
            y_batch = tf.squeeze(y_batch)
            out = model(x_raw)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            if preds.size()[0] > 1:
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()
        acc = correct / total
        acc_history.append(acc)
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
        if acc > best_acc:
            trials = 0
            best_acc = acc
            torch.save(model.state_dict(), 'best.pth')
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
    bestacc.append(best_acc)
