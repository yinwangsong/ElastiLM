import os
os.system('export HF_HOME=/data/share')
os.system('export HF_ENDPOINT=https://hf-mirror.com')

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import copy
import torch.nn.functional as F
from tqdm import tqdm
import json

import scores

import numpy as np
import random
import os

import sys
import os

# 获取当前文件的路径
current_path = os.path.dirname(__file__)

# 获取两级父目录的路径
parent_path = os.path.abspath(os.path.join(current_path, '../../../LLMPruner/'))

# 将父目录路径添加到sys.path中
if parent_path not in sys.path:
    sys.path.append(parent_path)

from LLMPruner.datasets.example_samples import get_examples

def set_seed(seed=42):
    """为所有可能的随机源设置种子以增加可重复性."""
    random.seed(seed)       # Python自带的random库
    np.random.seed(seed)    # Numpy库
    torch.manual_seed(seed) # CPU上的PyTorch种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)         # 为所有CUDA设备设置种子
        torch.cuda.manual_seed_all(seed)     # 为所有CUDA设备设置种子（如果有多个GPU）
        torch.backends.cudnn.deterministic = True  # 使用确定性算法，可能会影响性能
        torch.backends.cudnn.benchmark = False     # 关闭优化（benchmarking），保证可重复性

# 使用这个函数设置种子
set_seed(42)

scores.sparse.PRUNE = False
scores.sparse.SPARSE = False
scores.sparse.GET_HIDDEN_STATES_ATTN = True

model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

print(model)

layer_num = len(model.model.layers)

layer_imp = [0 for _ in range(layer_num)]

text1 = get_examples('bookcorpus', tokenizer, 10, seq_len = 64).cuda()

with torch.no_grad():
    out1 = model(text1, labels=text1)

for layer_idx in tqdm(range(layer_num)):

    hidden_states1 = scores.Hidden.hidden_states_attention[layer_idx*2]
    hidden_states2 = scores.Hidden.hidden_states_attention[layer_idx*2+1]
    
    layer_imp[layer_idx] += 1- torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)).item()

print(layer_imp)

def sorted_indices(lst):
    return [index for index, value in sorted(enumerate(lst), key=lambda x: x[1])]

print(sorted_indices(layer_imp)) 
# [22, 3, 20, 2, 26, 4, 18, 16, 5, 24, 27, 28, 30, 6, 29, 21, 13, 10, 15, 19, 8, 7, 12, 23, 25, 9, 14, 11, 17, 31, 0, 1]