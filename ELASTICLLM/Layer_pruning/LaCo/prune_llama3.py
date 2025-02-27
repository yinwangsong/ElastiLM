import numpy as np
import random
import os

import sys
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
# import scores

# 获取当前文件的路径
current_path = os.path.dirname(__file__)

# 获取两级父目录的路径
parent_path = os.path.abspath(os.path.join(current_path, '../../LLMPruner/'))

# 将父目录路径添加到sys.path中
if parent_path not in sys.path:
    sys.path.append(parent_path)


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

# scores.sparse.PRUNE = False
# scores.sparse.SPARSE = False

model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).to('cpu')
tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")

print(model)

INTERVAL = 1
MERGE_LAYERS = 4
HIGHEST_LAY = 31
LOWEST_LAY = 0
THRESHOLD = 0.45

NUMBER_LAYERS = 32

ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# cpu_models = []

import scores
scores.sparse.LACO = True

for RATIO in ratios:
    REVERVE_LAYERS_NUM = int(RATIO*NUMBER_LAYERS)
    from copy import deepcopy
    def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
        
        print(merge_layer_num, len(model.model.layers), merge_base_lay - 1)   
        merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
        print(merge_layer_num)
        
        model_copy = deepcopy(model)
        for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):      
            # gate_proj
            model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
                model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
            )
            # down_proj
            model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
                model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
            )
            # up_proj
            model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
                model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
            )
            

            # q_proj
            model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
                model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
            )

            # k_proj
            model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
                model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
            ) 
        
            # v_proj
            model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
                model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
            )
        
            # o_proj
            model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
                model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
            )

                        
        for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):

            del(model_copy.model.layers[diff_lay])
        return model_copy

    def cal_last_hidden_sim(model1, model2, tokenizer, sents):
        sim_ls = []
        for s in sents:
            # print(s)
            encoded_inputs = tokenizer(s, return_tensors='pt')
            encoded_inputs = {k: v.to('cuda:1') for k, v in encoded_inputs.items()}
            model1 = model1.to('cuda:1')
            with torch.no_grad():
                outputs1 = model1(**encoded_inputs, output_hidden_states=True)
            hidden_states1 = outputs1.hidden_states[-1].cpu() # (1, seq_len, hidden)
            encoded_inputs = {k: v.to('cuda:0') for k, v in encoded_inputs.items()}
            model2 = model2.to('cuda:0')
            # print(hidden_states1)
            with torch.no_grad():
                outputs2 = model2(**encoded_inputs, output_hidden_states=True)
            hidden_states2 = outputs2.hidden_states[-1].cpu() # (1, seq_len, hidden)
            sim_ls.append(torch.cosine_similarity(hidden_states1.squeeze(0).flatten().unsqueeze(0), hidden_states2.squeeze(0).flatten().unsqueeze(0)))
        sim_ls = [i.item() for i in sim_ls]
        print(sim_ls, np.mean(sim_ls))
        return np.mean(sim_ls)


    import copy
    llama_model_copy_to_compress = copy.deepcopy(model.cpu())

    lay = HIGHEST_LAY - MERGE_LAYERS
    last_merge_flag = False

    sents = []
    en_wiki_selected = ['Mouron () is a commune in the Arde',
    'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
    'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
    'Torreorgaz is a municipality in the',
    'Copa Libertadores 1973 was won by defending champions Independiente of A']

    # zh_wiki_selected = ['月桃   \xa0\xa0月桃月桃属草本，单叶，互生，具',
    #  '法国立贝尔洁白牙贴  目录产品成份：产品功效：用法用量：注意事项：产品禁忌：不良反应：规\xa0 \xa0 格：医疗器械注册号：产品执行标准：生产许可证号：授权监制：生产企业：',
    #  'TIMKEN 641/632-B轴承  目录TIMK',
    #  '天然碳化物质微结构研究  目录图书信息内容简介  图书信息作\u3000\u3000者： 冯有利 著 \n丛 书 名：\xa0\xa0出 版 社： 地质出版社 ISBN：9787116059771 出版时间',
    #  'V字领衣服  目录基本信息']

    sents.extend(en_wiki_selected)
    # sents.extend(zh_wiki_selected)


    while len(llama_model_copy_to_compress.model.layers) > REVERVE_LAYERS_NUM:
        if lay == LOWEST_LAY:
            THRESHOLD -= 0.1
        # print(lay)
        # print('current model layer', len(llama_model_copy_to_compress.model.layers))
        tmp_merged_model = merge_layers_return_model(llama_model_copy_to_compress.cpu(), lay, MERGE_LAYERS-1)
        # print("merge_layers_return_model done")
        sim_value = cal_last_hidden_sim(model, tmp_merged_model, tokenizer, sents)
        if sim_value > THRESHOLD:
            llama_model_copy_to_compress = tmp_merged_model
            lay -= INTERVAL
            if lay >= len(llama_model_copy_to_compress.model.layers):
                lay = len(llama_model_copy_to_compress.model.layers) - 1 - MERGE_LAYERS
        else:
            lay -= 1
        if lay < LOWEST_LAY:
            lay = LOWEST_LAY
    llama_model_copy_to_compress.config.num_hidden_layers = len(llama_model_copy_to_compress.model.layers)

    print(llama_model_copy_to_compress)
    llama_model_copy_to_compress = llama_model_copy_to_compress.cpu()
    torch.save(llama_model_copy_to_compress, "prune_log/LaCo/llama3_{}.pt".format(RATIO))