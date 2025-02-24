# import os
# os.system('export HF_HOME=/data/share')
# os.system('export HF_ENDPOINT=https://hf-mirror.com')

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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
import numpy

from auto_gptq.eval_tasks import SequenceClassificationTask, TextSummarizationTask, sequence_classification_task

import argparse

# datasets that opt can handle
# truthfulQA BoolQ LAMBADA SciQ octopus ARC-E


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

parser = argparse.ArgumentParser()
parser.add_argument('--mode') # Off-The-Shelf/Lingua2+Contextual/LLMPruner/LayerReduction/Ours
parser.add_argument('--prune_ratio', type=str, default="2.7b")
parser.add_argument('--prefill_SLO', type=float, default=1.0)
parser.add_argument('--decode_SLO', type=float, default=1.0)
parser.add_argument('--model')
parser.add_argument('--res_save_pth')

args = parser.parse_args()

PREFILL_SLO = [
    '[02]',
    '[03]',
    '[04]',
    '[05]',
    '[06]',
    '[07]',
    '[08]',
    '[09]'
]

DECODE_SLO = [
    '<05>',
    '<06>',
    '<07>',
    '<08>',
    '<09>',
    '<10>'
]

prefill_dict = {0.2: '[02]', 0.3: '[03]', 0.4: '[04]', 0.5: '[05]', 0.6: '[06]', 0.7: '[07]', 0.8: '[08]', 0.9: '[09]',}
decode_dict = {0.5: '<05>', 0.6: '<06>', 0.7: '<07>', 0.8: '<08>', 0.9: '<09>', 1.0: '<10>',}

prompt_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
model_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

prompt_ratios_dict = {1.0: 0, 0.9: 1, 0.8: 2, 0.7: 3, 0.6: 4, 0.5: 5, 0.4: 6, 0.3: 7, 0.2: 8, 0.1: 9}
model_ratios_dict = {0.2: 0, 0.3: 1, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 6, 0.9: 7, 1.0: 8}

class MultiTaskBertforSeqCLS(torch.nn.Module):
    def __init__(self, model, num_labels_task1, num_labels_task2):
        super().__init__()
        self.bert = model
        self.classifier1 = torch.nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier2 = torch.nn.Linear(self.bert.config.hidden_size, num_labels_task2)

    def forward(self, input_ids):
        outputs = self.bert(input_ids=input_ids)
        pooled_output = outputs.pooler_output
        logits_task1 = self.classifier1(pooled_output)
        logits_task2 = self.classifier2(pooled_output)
        return logits_task1, logits_task2

def reorder_llama(model):
    import scores, os
    directory = "./ELASTICLLM/imp/Ours/llama/llama_{pth}/rank_all.json".format(pth="0.2")
    with open(directory, 'r') as file:
        reoder_indices = json.load(file)

    def reorder_weights(weights, indices):
        # print(weights.shape, weights[indices].shape)
        return weights[indices]
    reoder_indices = [i[::-1].copy() for i in reoder_indices]

    index = 0
    layer_idx = 0
    layer_idx_all = 0
    for name, param in model.named_parameters():
        # print(name)
        if index%9==0 and index>0:
            layer_idx_all +=1
            if layer_idx_all not in [7, 4, 1, 0, 5, 31]:
                layer_idx += 1
        if layer_idx_all in [7, 4, 1, 0, 5, 31]:
            index += 1
            continue
        if 'self_attn' in name:
            indices =[]
            # print(layer_idx)
            for i in reoder_indices[(layer_idx-1)*2]:
                for j in range(128):
                    indices.append(i*128+j)
            if 'o_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), indices)
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights

        if 'mlp' in name:
            if 'down_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights

        index += 1

    for prune_ratio in model_ratios[:-1]:
        print(prune_ratio)
        import scores, os
        directory = "./ELASTICLLM/imp/Ours/llama/llama_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        # reorder lora
        lora = torch.load("/data/yinwangsong/ELASTICLLM/tune_log/llama_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama/llama_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama/llama_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        reoder_indices_core = []
        for i in range(len(pruned_indices)):
            if i%2==0:
                core_indices = sorted(list(set([_ for _ in range(4096)])-set(pruned_indices[i])))
                core_indices2 = []
                for j in core_indices:
                    if j%128==0:
                        core_indices2.append(int(j/128))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices2]
                    )[::-1].copy()
                )
            else:
                core_indices = sorted(list(set([_ for _ in range(11008)])-set(pruned_indices[i])))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices]
                    )[::-1].copy()
                )

        import scores
        index = 0
        layer_idx = 0
        layer_idx_all = 0
        for name in lora:
            param = lora[name].cuda()
            if index%14==0 and index>0:
                layer_idx_all +=1
                if layer_idx_all not in [7, 4, 1, 0, 5, 31]:
                    layer_idx += 1
            if layer_idx_all in [7, 4, 1, 0, 5, 31]:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 6:
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(param.transpose(0, 1))
                index += 1
            else:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 6:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param.transpose(0, 1), indices)
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(new_weights)
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 10:
                    new_weights = reorder_weights(param.transpose(0, 1), reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(new_weights)
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(new_weights.transpose(0, 1))
                index += 1
    return model

def get_llama(model, ratio):
    import scores
    # print(len(scores.mask.mlp_mask))
    scores.mask.head_mask = []
    scores.mask.mlp_mask = []
    if ratio == 1.0:
        import scores
        scores.sparse.PRUNE = False
        scores.sparse.LORA = False
        scores.sparse.SPARSE = False
        return model
    prune_ratio = ratio

    import scores
    scores.sparse.PRUNE = False
    scores.sparse.LORA = True
    scores.sparse.SPARSE = True


    del scores.LoRAs.lora_a_q
    del scores.LoRAs.lora_b_q
    del scores.LoRAs.lora_a_k
    del scores.LoRAs.lora_b_k
    del scores.LoRAs.lora_a_v
    del scores.LoRAs.lora_b_v
    del scores.LoRAs.lora_a_o
    del scores.LoRAs.lora_b_o
    del scores.LoRAs.lora_a_gate
    del scores.LoRAs.lora_b_gate
    del scores.LoRAs.lora_a_down
    del scores.LoRAs.lora_b_down
    del scores.LoRAs.lora_a_up
    del scores.LoRAs.lora_b_up


    scores.LoRAs.lora_a_q = scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio]
    scores.LoRAs.lora_b_q = scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio]
    scores.LoRAs.lora_a_k = scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio]
    scores.LoRAs.lora_b_k = scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio]
    scores.LoRAs.lora_a_v = scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio]
    scores.LoRAs.lora_b_v = scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio]
    scores.LoRAs.lora_a_o = scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio]
    scores.LoRAs.lora_b_o = scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio]
    scores.LoRAs.lora_a_gate = scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio]
    scores.LoRAs.lora_b_gate = scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio]
    scores.LoRAs.lora_a_down = scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio]
    scores.LoRAs.lora_b_down = scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio]
    scores.LoRAs.lora_a_up = scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio]
    scores.LoRAs.lora_b_up = scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio]
    
    # 准备mask
    import scores, os
    directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama/llama_{pth}/rank_pruned.json".format(pth=prune_ratio)
    with open(directory, 'r') as file:
        pruned_indices = json.load(file)

    real_idx = 0 
    for layer_idx in range(32):
        if layer_idx in [7, 4, 1, 0, 5, 31]:
            attn_mask = torch.ones((1, 32, 1, 128)).cuda()
            scores.mask.head_mask.append(attn_mask)
            mlp_mask = torch.ones((1, 1, 11008)).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
        else:
            # print(layer_idx, 1, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(32-len(pruned_indices[real_idx*2])/128)
            attn_mask = torch.cat(
                (
                    torch.ones((1, reserved_nums, 1, 128)),
                    torch.zeros((1, 32-reserved_nums, 1, 128))
                ),
                dim=1
            ).cuda()
            scores.mask.head_mask.append(attn_mask)
            # print(layer_idx, 2, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(11008-len(pruned_indices[real_idx*2+1]))
            mlp_mask = torch.cat(
                (
                    torch.ones((1, 1, reserved_nums)),
                    torch.zeros((1, 1, 11008-reserved_nums))
                ),
                dim=-1
            ).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
            # print(layer_idx, 3, torch.cuda.memory_reserved()/1024/1024)
            real_idx += 1
    return model

def reorder_vicuna(model):
    import scores, os
    directory = "./ELASTICLLM/imp/Ours/vicuna/vicuna_{pth}/rank_all.json".format(pth="0.2")
    with open(directory, 'r') as file:
        reoder_indices = json.load(file)

    def reorder_weights(weights, indices):
        # print(weights.shape, weights[indices].shape)
        return weights[indices]
    reoder_indices = [i[::-1].copy() for i in reoder_indices]

    index = 0
    layer_idx = 0
    layer_idx_all = 0
    for name, param in model.named_parameters():
        # print(name)
        if index%9==0 and index>0:
            layer_idx_all +=1
            if layer_idx_all not in [4, 3, 2, 31, 1, 0]:
                layer_idx += 1
        if layer_idx_all in [4, 3, 2, 31, 1, 0]:
            index += 1
            continue
        if 'self_attn' in name:
            indices =[]
            # print(layer_idx)
            for i in reoder_indices[(layer_idx-1)*2]:
                for j in range(128):
                    indices.append(i*128+j)
            if 'o_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), indices)
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights

        if 'mlp' in name:
            if 'down_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights

        index += 1

    for prune_ratio in model_ratios[:-1]:
        print(prune_ratio)
        import scores, os
        directory = "./ELASTICLLM/imp/Ours/vicuna/vicuna_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        # reorder lora
        lora = torch.load("/data/yinwangsong/ELASTICLLM/tune_log/vicuna_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/vicuna/vicuna_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/vicuna/vicuna_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        reoder_indices_core = []
        for i in range(len(pruned_indices)):
            if i%2==0:
                core_indices = sorted(list(set([_ for _ in range(4096)])-set(pruned_indices[i])))
                core_indices2 = []
                for j in core_indices:
                    if j%128==0:
                        core_indices2.append(int(j/128))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices2]
                    )[::-1].copy()
                )
            else:
                core_indices = sorted(list(set([_ for _ in range(11008)])-set(pruned_indices[i])))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices]
                    )[::-1].copy()
                )

        import scores
        index = 0
        layer_idx = 0
        layer_idx_all = 0
        for name in lora:
            param = lora[name].cuda()
            if index%14==0 and index>0:
                layer_idx_all +=1
                if layer_idx_all not in [4, 3, 2, 31, 1, 0]:
                    layer_idx += 1
            if layer_idx_all in [4, 3, 2, 31, 1, 0]:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 6:
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(param.transpose(0, 1))
                index += 1
            else:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 6:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param.transpose(0, 1), indices)
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(new_weights)
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 10:
                    new_weights = reorder_weights(param.transpose(0, 1), reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(new_weights)
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(new_weights.transpose(0, 1))
                index += 1
    return model

def get_vicuna(model, ratio):
    import scores
    # print(len(scores.mask.mlp_mask))
    scores.mask.head_mask = []
    scores.mask.mlp_mask = []
    if ratio == 1.0:
        import scores
        scores.sparse.PRUNE = False
        scores.sparse.LORA = False
        scores.sparse.SPARSE = False
        return model
    prune_ratio = ratio

    import scores
    scores.sparse.PRUNE = False
    scores.sparse.LORA = True
    scores.sparse.SPARSE = True


    del scores.LoRAs.lora_a_q
    del scores.LoRAs.lora_b_q
    del scores.LoRAs.lora_a_k
    del scores.LoRAs.lora_b_k
    del scores.LoRAs.lora_a_v
    del scores.LoRAs.lora_b_v
    del scores.LoRAs.lora_a_o
    del scores.LoRAs.lora_b_o
    del scores.LoRAs.lora_a_gate
    del scores.LoRAs.lora_b_gate
    del scores.LoRAs.lora_a_down
    del scores.LoRAs.lora_b_down
    del scores.LoRAs.lora_a_up
    del scores.LoRAs.lora_b_up


    scores.LoRAs.lora_a_q = scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio]
    scores.LoRAs.lora_b_q = scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio]
    scores.LoRAs.lora_a_k = scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio]
    scores.LoRAs.lora_b_k = scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio]
    scores.LoRAs.lora_a_v = scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio]
    scores.LoRAs.lora_b_v = scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio]
    scores.LoRAs.lora_a_o = scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio]
    scores.LoRAs.lora_b_o = scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio]
    scores.LoRAs.lora_a_gate = scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio]
    scores.LoRAs.lora_b_gate = scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio]
    scores.LoRAs.lora_a_down = scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio]
    scores.LoRAs.lora_b_down = scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio]
    scores.LoRAs.lora_a_up = scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio]
    scores.LoRAs.lora_b_up = scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio]
    
    # 准备mask
    import scores, os
    directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/vicuna/vicuna_{pth}/rank_pruned.json".format(pth=prune_ratio)
    with open(directory, 'r') as file:
        pruned_indices = json.load(file)

    real_idx = 0 
    for layer_idx in range(32):
        if layer_idx in [4, 3, 2, 31, 1, 0]:
            attn_mask = torch.ones((1, 32, 1, 128)).cuda()
            scores.mask.head_mask.append(attn_mask)
            mlp_mask = torch.ones((1, 1, 11008)).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
        else:
            # print(layer_idx, 1, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(32-len(pruned_indices[real_idx*2])/128)
            attn_mask = torch.cat(
                (
                    torch.ones((1, reserved_nums, 1, 128)),
                    torch.zeros((1, 32-reserved_nums, 1, 128))
                ),
                dim=1
            ).cuda()
            scores.mask.head_mask.append(attn_mask)
            # print(layer_idx, 2, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(11008-len(pruned_indices[real_idx*2+1]))
            mlp_mask = torch.cat(
                (
                    torch.ones((1, 1, reserved_nums)),
                    torch.zeros((1, 1, 11008-reserved_nums))
                ),
                dim=-1
            ).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
            # print(layer_idx, 3, torch.cuda.memory_reserved()/1024/1024)
            real_idx += 1
    return model

def reorder_orcamini(model):
    import scores, os
    directory = "./ELASTICLLM/imp/Ours/orca_mini/orca_mini_{pth}/rank_all.json".format(pth="0.2")
    with open(directory, 'r') as file:
        reoder_indices = json.load(file)

    def reorder_weights(weights, indices):
        # print(weights.shape, weights[indices].shape)
        return weights[indices]
    reoder_indices = [i[::-1].copy() for i in reoder_indices]

    index = 0
    layer_idx = 0
    layer_idx_all = 0
    for name, param in model.named_parameters():
        # print(name)
        if index%9==0 and index>0:
            layer_idx_all +=1
            if layer_idx_all not in [13, 6, 1, 25, 2, 0]:
                layer_idx += 1
        if layer_idx_all in [13, 6, 1, 25, 2, 0]:
            index += 1
            continue
        if 'self_attn' in name:
            indices =[]
            # print(layer_idx)
            for i in reoder_indices[(layer_idx-1)*2]:
                for j in range(100):
                    indices.append(i*100+j)
            if 'o_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), indices)
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights

        if 'mlp' in name:
            if 'down_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights

        index += 1

    for prune_ratio in model_ratios[:-1]:
        print(prune_ratio)
        import scores, os
        directory = "./ELASTICLLM/imp/Ours/orca_mini/orca_mini_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        # reorder lora
        lora = torch.load("/data/yinwangsong/ELASTICLLM/tune_log/orca_mini_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/orca_mini/orca_mini_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/orca_mini/orca_mini_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        reoder_indices_core = []
        for i in range(len(pruned_indices)):
            if i%2==0:
                core_indices = sorted(list(set([_ for _ in range(3200)])-set(pruned_indices[i])))
                core_indices2 = []
                for j in core_indices:
                    if j%100==0:
                        core_indices2.append(int(j/100))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices2]
                    )[::-1].copy()
                )
            else:
                core_indices = sorted(list(set([_ for _ in range(8640)])-set(pruned_indices[i])))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices]
                    )[::-1].copy()
                )

        import scores
        index = 0
        layer_idx = 0
        layer_idx_all = 0
        for name in lora:
            param = lora[name].cuda()
            if index%14==0 and index>0:
                layer_idx_all +=1
                if layer_idx_all not in [13, 6, 1, 25, 2, 0]:
                    layer_idx += 1
            if layer_idx_all in [13, 6, 1, 25, 2, 0]:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 6:
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(param.transpose(0, 1))
                index += 1
            else:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(100):
                            indices.append(i*100+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(100):
                            indices.append(i*100+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(100):
                            indices.append(i*100+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 6:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(100):
                            indices.append(i*100+j)
                    new_weights = reorder_weights(param.transpose(0, 1), indices)
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(new_weights)
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 10:
                    new_weights = reorder_weights(param.transpose(0, 1), reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(new_weights)
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(new_weights.transpose(0, 1))
                index += 1
    return model

def get_orcamini(model, ratio):
    import scores
    # print(len(scores.mask.mlp_mask))
    scores.mask.head_mask = []
    scores.mask.mlp_mask = []
    if ratio == 1.0:
        import scores
        scores.sparse.PRUNE = False
        scores.sparse.LORA = False
        scores.sparse.SPARSE = False
        return model
    prune_ratio = ratio

    import scores
    scores.sparse.PRUNE = False
    scores.sparse.LORA = True
    scores.sparse.SPARSE = True


    del scores.LoRAs.lora_a_q
    del scores.LoRAs.lora_b_q
    del scores.LoRAs.lora_a_k
    del scores.LoRAs.lora_b_k
    del scores.LoRAs.lora_a_v
    del scores.LoRAs.lora_b_v
    del scores.LoRAs.lora_a_o
    del scores.LoRAs.lora_b_o
    del scores.LoRAs.lora_a_gate
    del scores.LoRAs.lora_b_gate
    del scores.LoRAs.lora_a_down
    del scores.LoRAs.lora_b_down
    del scores.LoRAs.lora_a_up
    del scores.LoRAs.lora_b_up


    scores.LoRAs.lora_a_q = scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio]
    scores.LoRAs.lora_b_q = scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio]
    scores.LoRAs.lora_a_k = scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio]
    scores.LoRAs.lora_b_k = scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio]
    scores.LoRAs.lora_a_v = scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio]
    scores.LoRAs.lora_b_v = scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio]
    scores.LoRAs.lora_a_o = scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio]
    scores.LoRAs.lora_b_o = scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio]
    scores.LoRAs.lora_a_gate = scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio]
    scores.LoRAs.lora_b_gate = scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio]
    scores.LoRAs.lora_a_down = scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio]
    scores.LoRAs.lora_b_down = scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio]
    scores.LoRAs.lora_a_up = scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio]
    scores.LoRAs.lora_b_up = scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio]
    
    # 准备mask
    import scores, os
    directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/orca_mini/orca_mini_{pth}/rank_pruned.json".format(pth=prune_ratio)
    with open(directory, 'r') as file:
        pruned_indices = json.load(file)

    real_idx = 0 
    for layer_idx in range(26):
        if layer_idx in [13, 6, 1, 25, 2, 0]:
            attn_mask = torch.ones((1, 32, 1, 100)).cuda()
            scores.mask.head_mask.append(attn_mask)
            mlp_mask = torch.ones((1, 1, 8640)).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
        else:
            # print(layer_idx, 1, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(32-len(pruned_indices[real_idx*2])/100)
            attn_mask = torch.cat(
                (
                    torch.ones((1, reserved_nums, 1, 100)),
                    torch.zeros((1, 32-reserved_nums, 1, 100))
                ),
                dim=1
            ).cuda()
            scores.mask.head_mask.append(attn_mask)
            # print(layer_idx, 2, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(8640-len(pruned_indices[real_idx*2+1]))
            mlp_mask = torch.cat(
                (
                    torch.ones((1, 1, reserved_nums)),
                    torch.zeros((1, 1, 8640-reserved_nums))
                ),
                dim=-1
            ).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
            # print(layer_idx, 3, torch.cuda.memory_reserved()/1024/1024)
            real_idx += 1
    return model

def reorder_llama3(model):
    import scores, os
    directory = "./ELASTICLLM/imp/Ours/llama3/llama3_{pth}/rank_all.json".format(pth="0.2")
    with open(directory, 'r') as file:
        reoder_indices = json.load(file)

    def reorder_weights(weights, indices):
        return weights[indices]
    reoder_indices = [i[::-1].copy() for i in reoder_indices]

    index = 0
    layer_idx = 0
    layer_idx_all = 0
    for name, param in model.named_parameters():
        # print(name)
        if index%9==0 and index>0:
            layer_idx_all +=1
            if layer_idx_all not in [29, 2, 30, 31, 1, 0]:
                layer_idx += 1
        if layer_idx_all in [29, 2, 30, 31, 1, 0]:
            index += 1
            continue
        if 'self_attn' in name:

            if 'o_proj' in name:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128*4):
                        indices.append(i*128*4+j)

                new_weights = reorder_weights(param.data.transpose(0, 1), indices)
                param.data = new_weights.transpose(0, 1)
            elif 'q_proj' in name:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128*4):
                        indices.append(i*128*4+j)
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights 
            else:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128):
                        indices.append(i*128+j)
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights

        if 'mlp' in name:
            if 'down_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights

        index += 1

    for prune_ratio in model_ratios[:-1]:
        print(prune_ratio)
        import scores, os
        directory = "./ELASTICLLM/imp/Ours/llama3/llama3_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        # reorder lora
        lora = torch.load("/data/yinwangsong/ELASTICLLM/tune_log/llama3_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3/llama3_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3/llama3_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        reoder_indices_core = []
        for i in range(len(pruned_indices)):
            if i%2==0:
                core_indices = sorted(list(set([_ for _ in range(4096//4)])-set(pruned_indices[i])))
                core_indices2 = []
                for j in core_indices:
                    if j%128==0:
                        core_indices2.append(int(j/128))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices2]
                    )[::-1].copy()
                )
            else:
                core_indices = sorted(list(set([_ for _ in range(14336)])-set(pruned_indices[i])))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices]
                    )[::-1].copy()
                )

        import scores
        index = 0
        layer_idx = 0
        layer_idx_all = 0
        for name in lora:
            param = lora[name].cuda()
            if index%14==0 and index>0:
                layer_idx_all +=1
                if layer_idx_all not in [29, 2, 30, 31, 1, 0]:
                    layer_idx += 1
            if layer_idx_all in [29, 2, 30, 31, 1, 0]:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 6:
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                index += 1
            else:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128*4):
                            indices.append(i*128*4+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 6:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128*4):
                            indices.append(i*128*4+j)
                    new_weights = reorder_weights(param.transpose(0, 1), indices)
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(new_weights)
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 12:
                    new_weights = reorder_weights(param.transpose(0, 1), reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(new_weights)
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                index += 1
    return model

def get_llama3(model, ratio):
    import scores
    # print(len(scores.mask.mlp_mask))
    scores.mask.head_mask = []
    scores.mask.mlp_mask = []
    if ratio == 1.0:
        import scores
        scores.sparse.PRUNE = False
        scores.sparse.LORA = False
        scores.sparse.SPARSE = False
        return model
    prune_ratio = ratio

    import scores
    scores.sparse.PRUNE = False
    scores.sparse.LORA = True
    scores.sparse.SPARSE = True


    del scores.LoRAs.lora_a_q
    del scores.LoRAs.lora_b_q
    del scores.LoRAs.lora_a_k
    del scores.LoRAs.lora_b_k
    del scores.LoRAs.lora_a_v
    del scores.LoRAs.lora_b_v
    del scores.LoRAs.lora_a_o
    del scores.LoRAs.lora_b_o
    del scores.LoRAs.lora_a_gate
    del scores.LoRAs.lora_b_gate
    del scores.LoRAs.lora_a_down
    del scores.LoRAs.lora_b_down
    del scores.LoRAs.lora_a_up
    del scores.LoRAs.lora_b_up


    scores.LoRAs.lora_a_q = scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio]
    scores.LoRAs.lora_b_q = scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio]
    scores.LoRAs.lora_a_k = scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio]
    scores.LoRAs.lora_b_k = scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio]
    scores.LoRAs.lora_a_v = scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio]
    scores.LoRAs.lora_b_v = scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio]
    scores.LoRAs.lora_a_o = scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio]
    scores.LoRAs.lora_b_o = scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio]
    scores.LoRAs.lora_a_gate = scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio]
    scores.LoRAs.lora_b_gate = scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio]
    scores.LoRAs.lora_a_down = scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio]
    scores.LoRAs.lora_b_down = scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio]
    scores.LoRAs.lora_a_up = scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio]
    scores.LoRAs.lora_b_up = scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio]
    
    # 准备mask
    import scores, os
    directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3/llama3_{pth}/rank_pruned.json".format(pth=prune_ratio)
    with open(directory, 'r') as file:
        pruned_indices = json.load(file)

    real_idx = 0 
    for layer_idx in range(32):
        if layer_idx in [29, 2, 30, 31, 1, 0]:
            attn_mask = torch.ones((1, 32, 1, 128)).cuda()
            scores.mask.head_mask.append(attn_mask)
            mlp_mask = torch.ones((1, 1, 14336)).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
        else:
            # print(layer_idx, 1, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(32-len(pruned_indices[real_idx*2])/128*4)
            attn_mask = torch.cat(
                (
                    torch.ones((1, reserved_nums, 1, 128)),
                    torch.zeros((1, 32-reserved_nums, 1, 128))
                ),
                dim=1
            ).cuda()
            scores.mask.head_mask.append(attn_mask)
            # print(layer_idx, 2, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(14336-len(pruned_indices[real_idx*2+1]))
            mlp_mask = torch.cat(
                (
                    torch.ones((1, 1, reserved_nums)),
                    torch.zeros((1, 1, 14336-reserved_nums))
                ),
                dim=-1
            ).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
            # print(layer_idx, 3, torch.cuda.memory_reserved()/1024/1024)
            real_idx += 1
    return model

def reorder_llama3_instruct(model):
    import scores, os
    directory = "./ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_all.json".format(pth="0.2")
    with open(directory, 'r') as file:
        reoder_indices = json.load(file)

    def reorder_weights(weights, indices):
        return weights[indices]
    reoder_indices = [i[::-1].copy() for i in reoder_indices]

    index = 0
    layer_idx = 0
    layer_idx_all = 0
    for name, param in model.named_parameters():
        # print(name)
        if index%9==0 and index>0:
            layer_idx_all +=1
            if layer_idx_all not in [27, 30, 2, 31, 1, 0]:
                layer_idx += 1
        if layer_idx_all in [27, 30, 2, 31, 1, 0]:
            index += 1
            continue
        if 'self_attn' in name:

            if 'o_proj' in name:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128*4):
                        indices.append(i*128*4+j)

                new_weights = reorder_weights(param.data.transpose(0, 1), indices)
                param.data = new_weights.transpose(0, 1)
            elif 'q_proj' in name:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128*4):
                        indices.append(i*128*4+j)
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights 
            else:
                indices =[]
                # print(layer_idx)
                for i in reoder_indices[(layer_idx-1)*2]:
                    for j in range(128):
                        indices.append(i*128+j)
                new_weights = reorder_weights(param.data, indices)
                param.data = new_weights

        if 'mlp' in name:
            if 'down_proj' in name:
                new_weights = reorder_weights(param.data.transpose(0, 1), reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights.transpose(0, 1)
            else:
                new_weights = reorder_weights(param.data, reoder_indices[(layer_idx-1)*2+1])
                param.data = new_weights

        index += 1

    for prune_ratio in model_ratios[:-1]:
        print(prune_ratio)
        import scores, os
        directory = "./ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        # reorder lora
        lora = torch.load("/data/yinwangsong/ELASTICLLM/tune_log/llama3_instruct_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_pruned.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            pruned_indices = json.load(file)

        reoder_indices_core = []
        for i in range(len(pruned_indices)):
            if i%2==0:
                core_indices = sorted(list(set([_ for _ in range(4096//4)])-set(pruned_indices[i])))
                core_indices2 = []
                for j in core_indices:
                    if j%128==0:
                        core_indices2.append(int(j/128))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices2]
                    )[::-1].copy()
                )
            else:
                core_indices = sorted(list(set([_ for _ in range(14336)])-set(pruned_indices[i])))
                reoder_indices_core.append(
                    numpy.argsort(
                        [imps[i][_] for _ in core_indices]
                    )[::-1].copy()
                )

        import scores
        index = 0
        layer_idx = 0
        layer_idx_all = 0
        for name in lora:
            param = lora[name].cuda()
            if index%14==0 and index>0:
                layer_idx_all +=1
                if layer_idx_all not in [27, 30, 2, 31, 1, 0]:
                    layer_idx += 1
            if layer_idx_all in [27, 30, 2, 31, 1, 0]:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 6:
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 12:
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                index += 1
            else:
                if index % 14 == 0:
                    scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 1:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128*4):
                            indices.append(i*128*4+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 2:
                    scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 3:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 4:
                    scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 5:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128):
                            indices.append(i*128+j)
                    new_weights = reorder_weights(param, indices)
                    scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 6:
                    indices =[]
                    for i in reoder_indices_core[(layer_idx-1)*2]:
                        for j in range(128*4):
                            indices.append(i*128*4+j)
                    new_weights = reorder_weights(param.transpose(0, 1), indices)
                    scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio].append(new_weights)
                if index % 14 == 7:
                    scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 8:
                    scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 9:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 10:
                    scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio].append(param.transpose(0, 1))
                if index % 14 == 11:
                    new_weights = reorder_weights(param, reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio].append(new_weights.transpose(0, 1))
                if index % 14 == 12:
                    new_weights = reorder_weights(param.transpose(0, 1), reoder_indices_core[(layer_idx-1)*2+1])
                    scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio].append(new_weights)
                if index % 14 == 13:
                    scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio].append(param.transpose(0, 1))
                index += 1
    return model

def get_llama3_instruct(model, ratio):
    import scores
    # print(len(scores.mask.mlp_mask))
    scores.mask.head_mask = []
    scores.mask.mlp_mask = []
    if ratio == 1.0:
        import scores
        scores.sparse.PRUNE = False
        scores.sparse.LORA = False
        scores.sparse.SPARSE = False
        return model
    prune_ratio = ratio

    import scores
    scores.sparse.PRUNE = False
    scores.sparse.LORA = True
    scores.sparse.SPARSE = True


    del scores.LoRAs.lora_a_q
    del scores.LoRAs.lora_b_q
    del scores.LoRAs.lora_a_k
    del scores.LoRAs.lora_b_k
    del scores.LoRAs.lora_a_v
    del scores.LoRAs.lora_b_v
    del scores.LoRAs.lora_a_o
    del scores.LoRAs.lora_b_o
    del scores.LoRAs.lora_a_gate
    del scores.LoRAs.lora_b_gate
    del scores.LoRAs.lora_a_down
    del scores.LoRAs.lora_b_down
    del scores.LoRAs.lora_a_up
    del scores.LoRAs.lora_b_up


    scores.LoRAs.lora_a_q = scores.LoRAs_all_in_Dict.lora_a_q[prune_ratio]
    scores.LoRAs.lora_b_q = scores.LoRAs_all_in_Dict.lora_b_q[prune_ratio]
    scores.LoRAs.lora_a_k = scores.LoRAs_all_in_Dict.lora_a_k[prune_ratio]
    scores.LoRAs.lora_b_k = scores.LoRAs_all_in_Dict.lora_b_k[prune_ratio]
    scores.LoRAs.lora_a_v = scores.LoRAs_all_in_Dict.lora_a_v[prune_ratio]
    scores.LoRAs.lora_b_v = scores.LoRAs_all_in_Dict.lora_b_v[prune_ratio]
    scores.LoRAs.lora_a_o = scores.LoRAs_all_in_Dict.lora_a_o[prune_ratio]
    scores.LoRAs.lora_b_o = scores.LoRAs_all_in_Dict.lora_b_o[prune_ratio]
    scores.LoRAs.lora_a_gate = scores.LoRAs_all_in_Dict.lora_a_gate[prune_ratio]
    scores.LoRAs.lora_b_gate = scores.LoRAs_all_in_Dict.lora_b_gate[prune_ratio]
    scores.LoRAs.lora_a_down = scores.LoRAs_all_in_Dict.lora_a_down[prune_ratio]
    scores.LoRAs.lora_b_down = scores.LoRAs_all_in_Dict.lora_b_down[prune_ratio]
    scores.LoRAs.lora_a_up = scores.LoRAs_all_in_Dict.lora_a_up[prune_ratio]
    scores.LoRAs.lora_b_up = scores.LoRAs_all_in_Dict.lora_b_up[prune_ratio]
    
    # 准备mask
    import scores, os
    directory = "/data/yinwangsong/ELASTICLLM/ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_pruned.json".format(pth=prune_ratio)
    with open(directory, 'r') as file:
        pruned_indices = json.load(file)

    real_idx = 0 
    for layer_idx in range(32):
        if layer_idx in [27, 30, 2, 31, 1, 0]:
            attn_mask = torch.ones((1, 32, 1, 128)).cuda()
            scores.mask.head_mask.append(attn_mask)
            mlp_mask = torch.ones((1, 1, 14336)).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
        else:
            # print(layer_idx, 1, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(32-len(pruned_indices[real_idx*2])/128*4)
            attn_mask = torch.cat(
                (
                    torch.ones((1, reserved_nums, 1, 128)),
                    torch.zeros((1, 32-reserved_nums, 1, 128))
                ),
                dim=1
            ).cuda()
            scores.mask.head_mask.append(attn_mask)
            # print(layer_idx, 2, torch.cuda.memory_reserved()/1024/1024)
            reserved_nums = int(14336-len(pruned_indices[real_idx*2+1]))
            mlp_mask = torch.cat(
                (
                    torch.ones((1, 1, reserved_nums)),
                    torch.zeros((1, 1, 14336-reserved_nums))
                ),
                dim=-1
            ).cuda()
            scores.mask.mlp_mask.append(mlp_mask)
            # print(layer_idx, 3, torch.cuda.memory_reserved()/1024/1024)
            real_idx += 1
    return model


if args.mode == "Off-The-Shelf":
    scores.sparse.PRUNE = False
    scores.sparse.SPARSE = False
    scores.sparse.LORA = False

    if args.prune_ratio == "2.7b":
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
    if args.prune_ratio == "1.3b":
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    if args.prune_ratio == "350m":
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

if args.mode == "Lingua2+Contextual":
    prompt_ratio = args.prefill_SLO
    model_ratio = args.decode_SLO
    if args.model == "llama":
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    if args.model == "vicuna":
        model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    if args.model == "orca3b-mini":
        model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")

    if args.model == "llama3":
        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")

    if args.model == "llama3-instruct":
        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct")

if args.mode == "LLMPruner":
    prune_ratio = args.prefill_SLO

    # reorder the model
    scores.sparse.PRUNE = False
    scores.sparse.SPARSE = True
    scores.sparse.LORA = False

    if args.model == "llama":
        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        model = reorder_llama(model)
        model = get_llama(model, prune_ratio)

    if args.model == "vicuna":
        model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        model = reorder_vicuna(model)
        model = get_vicuna(model, prune_ratio)

    if args.model == "orca3b-mini":

        model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")
        model = reorder_orcamini(model)
        model = get_orcamini(model, prune_ratio)

    if args.model == "llama3":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")
        model = reorder_llama3(model)
        model = get_llama3(model, prune_ratio)


    if args.model == "llama3-instruct":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct")
        model = reorder_llama3_instruct(model)
        model = get_llama3_instruct(model, prune_ratio)

if args.mode == "LayerReduction":
    if args.model == "llama":

        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")


        layer_idx_ranks = [24, 28, 27, 22, 11, 17, 6, 26, 2, 21, 13, 19, 12, 15, 20, 23, 18, 14, 29, 25, 10, 3, 9, 16, 8, 30, 7, 4, 1, 0, 5, 31]
    
        prune_ratio = min(args.prefill_SLO, args.decode_SLO)

        layer_idx_retain = layer_idx_ranks[int(1-float(prune_ratio)*32):]

        for i in range(32):
            if i in layer_idx_retain:
                model.model.layers[i].is_pruned = False
            else:
                model.model.layers[i].is_pruned = True

    if args.model == "llama3":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")


        layer_idx_ranks = [8, 26, 24, 23, 9, 12, 25, 19, 22, 13, 17, 21, 11, 10, 14, 20, 4, 15, 28, 6, 3, 18, 7, 16, 5, 27, 29, 2, 30, 31, 1, 0]
    
        prune_ratio = min(args.prefill_SLO, args.decode_SLO)

        layer_idx_retain = layer_idx_ranks[int(1-float(prune_ratio)*32):]

        for i in range(32):
            if i in layer_idx_retain:
                model.model.layers[i].is_pruned = False
            else:
                model.model.layers[i].is_pruned = True

    if args.model == "llama3-instruct":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct")


        layer_idx_ranks = [26, 13, 25, 23, 8, 21, 12, 19, 15, 24, 10, 22, 17, 9, 11, 28, 14, 18, 20, 3, 7, 4, 16, 6, 5, 29, 27, 30, 2, 31, 1, 0]
    
        prune_ratio = min(args.prefill_SLO, args.decode_SLO)

        layer_idx_retain = layer_idx_ranks[int(1-float(prune_ratio)*32):]

        for i in range(32):
            if i in layer_idx_retain:
                model.model.layers[i].is_pruned = False
            else:
                model.model.layers[i].is_pruned = True

    if args.model == "vicuna":

        model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")


        layer_idx_ranks = [12, 28, 7, 20, 29, 22, 10, 24, 27, 8, 30, 15, 6, 9, 26, 21, 18, 11, 25, 23, 5, 17, 16, 13, 14, 19, 4, 3, 2, 31, 1, 0]
    
        prune_ratio = min(args.prefill_SLO, args.decode_SLO)

        layer_idx_retain = layer_idx_ranks[int(1-float(prune_ratio)*32):]

        for i in range(32):
            if i in layer_idx_retain:
                model.model.layers[i].is_pruned = False
            else:
                model.model.layers[i].is_pruned = True
    if args.model == "orca3b-mini":

        model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")


        layer_idx_ranks = [15, 14, 19, 12, 21, 22, 10, 8, 11, 3, 23, 17, 7, 9, 24, 5, 16, 18, 4, 20, 13, 6, 1, 25, 2, 0]
    
        prune_ratio = min(args.prefill_SLO, args.decode_SLO)

        layer_idx_retain = layer_idx_ranks[int(1-float(prune_ratio)*26):]

        for i in range(26):
            if i in layer_idx_retain:
                model.model.layers[i].is_pruned = False
            else:
                model.model.layers[i].is_pruned = True

if args.mode == "Ours":

    # currently, only 
    prune_ratio = args.decode_SLO
    prompt_compress_ratio = round(float(args.prefill_SLO)/float(args.decode_SLO), 1)

    # reorder the model
    scores.sparse.PRUNE = False
    scores.sparse.SPARSE = True
    scores.sparse.LORA = True

    if args.model == "llama":

        model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        model = reorder_llama(model)
        model = get_llama(model, prune_ratio)

        decision_head = torch.load("ELASTICLLM/train_slm/{}/slm_decisionhead_llama.pt".format(args.model)).cuda()
        decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        slm = torch.load("ELASTICLLM/train_slm/{}/slm_scorehead.pt".format(args.model)).cuda()

    if args.model == "vicuna":

        model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        model = reorder_vicuna(model)
        model = get_vicuna(model, prune_ratio)

        decision_head = torch.load("ELASTICLLM/train_slm/{}/slm_decisionhead_vicuna.pt".format(args.model)).cuda()
        # decision_head = torch.load("ELASTICLLM/train_slm/llama/slm_decisionhead_llama.pt").cuda()
        decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        slm = torch.load("ELASTICLLM/train_slm/{}/slm_scorehead.pt".format(args.model)).cuda()
        # slm = torch.load("ELASTICLLM/train_slm/llama/slm_scorehead.pt").cuda()

    if args.model == "orca3b-mini":

        model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")
        model = reorder_orcamini(model)
        model = get_orcamini(model, prune_ratio)

        decision_head = torch.load("ELASTICLLM/train_slm/orcamini/slm_decisionhead_orcamini.pt").cuda()
        # decision_head = torch.load("ELASTICLLM/train_slm/llama/slm_decisionhead_llama.pt").cuda()
        decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        slm = torch.load("ELASTICLLM/train_slm/orcamini/slm_scorehead.pt").cuda()
        # slm = torch.load("ELASTICLLM/train_slm/llama/slm_scorehead.pt").cuda()


    if args.model == "llama3":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")
        model = reorder_llama3(model)
        model = get_llama3(model, prune_ratio)

        decision_head = torch.load("ELASTICLLM/train_slm/{}/slm_decisionhead_llama3.pt".format(args.model)).cuda()
        # decision_head = torch.load("ELASTICLLM/train_slm/llama/slm_decisionhead_llama.pt").cuda()
        decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        slm = torch.load("ELASTICLLM/train_slm/{}/slm_scorehead.pt".format(args.model)).cuda()
        # slm = torch.load("ELASTICLLM/train_slm/llama/slm_scorehead.pt").cuda()

    if args.model == "llama3-instruct":

        model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct")
        model = reorder_llama3_instruct(model)
        model = get_llama3_instruct(model, prune_ratio)

        decision_head = torch.load("ELASTICLLM/train_slm/llama3_instruct/slm_decisionhead_llama3_instruct.pt").cuda()
        # decision_head = torch.load("ELASTICLLM/train_slm/llama/slm_decisionhead_llama.pt").cuda()
        decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        slm = torch.load("ELASTICLLM/train_slm/llama3_instruct/slm_scorehead.pt").cuda()
        # slm = torch.load("ELASTICLLM/train_slm/llama/slm_scorehead.pt").cuda()

if args.mode == "debug":
    # scores.sparse.PRUNE = False
    # scores.sparse.SPARSE = False
    # scores.sparse.LORA = False
    
    import sys

    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    # # 获取当前文件的路径
    # current_path = os.path.dirname(__file__)

    # # 获取两级父目录的路径
    # parent_path = os.path.abspath(os.path.join(current_path, '../LLMPruner/'))

    # # 将父目录路径添加到sys.path中
    # if parent_path not in sys.path:
    #     sys.path.append(parent_path)
    
    # from LLMPruner.peft import PeftModel


    # pruned_dict = torch.load('/data/yinwangsong/LLM-Pruner/prune_log/prune_ckpt_path/pytorch_model.bin', map_location='cuda')
    # tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model'].half()

    # model = PeftModel.from_pretrained(
    #     model,
    #     "/data/yinwangsong/LLM-Pruner/tune_log/llama_0.5",
    #     map_location='cuda'
    # ).half()
if args.mode == "permute_submodels_recovered":
    prune_ratio = args.prefill_SLO

    import sys

    # 获取当前文件的路径
    current_path = os.path.dirname(__file__)

    # 获取两级父目录的路径
    parent_path = os.path.abspath(os.path.join(current_path, '../LLMPruner/'))

    # 将父目录路径添加到sys.path中
    if parent_path not in sys.path:
        sys.path.append(parent_path)
    
    from LLMPruner.peft import PeftModel

    pruned_dict = torch.load("prune_log/llama_{}/pytorch_model.bin".format(prune_ratio), map_location='cuda')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model'].half()

    model = PeftModel.from_pretrained(
        model,
        "tune_log/llama_{}/".format(prune_ratio),
        map_location='cuda'
    ).half()

if args.mode == "llmpruner_submodels":
    scores.sparse.PRUNE = False
    scores.sparse.SPARSE = False
    scores.sparse.LORA = False

    prune_ratio = args.prefill_SLO

    import sys

    # 获取当前文件的路径
    current_path = os.path.dirname(__file__)

    # 获取两级父目录的路径
    parent_path = os.path.abspath(os.path.join(current_path, '../LLMPruner/'))

    # 将父目录路径添加到sys.path中
    if parent_path not in sys.path:
        sys.path.append(parent_path)

    model = torch.load("prune_log/llama_{}_alllayers/pytorch_model.bin".format(prune_ratio))["model"].half().cuda()
    tokenizer = torch.load("prune_log/llama_{}_alllayers/pytorch_model.bin".format(prune_ratio))["tokenizer"]

if args.mode == "llmpruner_submodels_recovered":

    prune_ratio = args.prefill_SLO

    import sys

    # 获取当前文件的路径
    current_path = os.path.dirname(__file__)

    # 获取两级父目录的路径
    parent_path = os.path.abspath(os.path.join(current_path, '../LLMPruner/'))

    # 将父目录路径添加到sys.path中
    if parent_path not in sys.path:
        sys.path.append(parent_path)
    
    from LLMPruner.peft import PeftModel

    pruned_dict = torch.load("prune_log/llama_{}_alllayers/pytorch_model.bin".format(prune_ratio), map_location='cuda')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model'].half()

    model = PeftModel.from_pretrained(
        model,
        "tune_log/llama_{}_alllayers/".format(prune_ratio),
        map_location='cuda'
    ).half()

tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

dataset = datasets.load_dataset(path="allenai/ai2_arc", name="ARC-Easy", split="validation")

print(dataset)

SYS_PROMPT = ""
SYS_PROMPT += "You are a smart assistant that helps human sloving problems. You help them by answering questions.\n"
SYS_PROMPT += "Examples: \nQuestion: Which factor will most likely cause a person to develop a fever? Answer: a bacterial population in the bloodstream."
SYS_PROMPT += "\nQuestion: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship? Answer: food."
SYS_PROMPT += "\nQuestion: When a switch is used in an electrical circuit, the switch can Answer: stop and start the flow of current."
SYS_PROMPT += "\nQuestion: Which of the following is an example of an assistive device? Answer: contact lens."
SYS_PROMPT += "\nQuestion: Rocks are classified as igneous, metamorphic, or sedimentary according to Answer: how they formed."

QUERY = "Question: {question} "
QUERY += "Answer:"
LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
LABELS_NUMBER = ['1', '2', '3', '4', '5', '6', '7', '8']

model = model.eval()

num_right = 0
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        
        # compress the prompt
        if args.mode == "Ours": 

            # decision_head = torch.load("ELASTICLLM/train_slm/{}/slm_decisionhead_llama.pt".format(args.model)).cuda()
            # decision_head_tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
            special_tokens_dict = {'additional_special_tokens': PREFILL_SLO + DECODE_SLO}
            num_added_toks = decision_head_tokenizer.add_special_tokens(special_tokens_dict)
            decision_text = prefill_dict[args.prefill_SLO] + " " + decode_dict[args.decode_SLO] + " " + SYS_PROMPT + QUERY
            
            decision_head_input = decision_head_tokenizer.encode(
                decision_text,
                return_tensors='pt'
            ).cuda()

            prompt_ratio, model_ratio = decision_head(decision_head_input)

            prompt_ratio = prompt_ratios[torch.argmax(prompt_ratio.squeeze()).item()]
            model_ratio = model_ratios[torch.argmax(model_ratio.squeeze()).item()]

            if prompt_ratio * model_ratio > args.prefill_SLO or model_ratio > args.decode_SLO:
                model_ratio = 1.0
                for r in model_ratios:
                    if prompt_ratio * r <= args.prefill_SLO and r <= args.decode_SLO:
                        model_ratio = r

            prompt_compress_ratio = prompt_ratio

            if args.model == "llama":
                model = get_llama(model, model_ratio)
            if args.model == "vicuna":
                model = get_vicuna(model, model_ratio)
            if args.model == "orca3b-mini":
                model = get_orcamini(model, model_ratio)
            if args.model == "llama3":
                model = get_llama3(model, model_ratio)
            if args.model == "llama3-instruct":
                model = get_llama3_instruct(model, model_ratio)

            max_query_len = 0
            for j in range(len(dataset[i]['choices']['text'])):
                NEW_QUERY = "Question: {question} Answer: {answer}".format(question=dataset[i]['question'], answer=dataset[i]['choices']['text'][j])
                query = tokenizer.encode(
                    NEW_QUERY
                )
                if len(query) > max_query_len:
                    max_query_len = len(query)
            sys_prompt_ids = tokenizer.encode(
                SYS_PROMPT
            )
            sys_prompt_len = len(sys_prompt_ids)
            sys_prompt_compress_ratio = ((sys_prompt_len + max_query_len) * prompt_compress_ratio - max_query_len) / sys_prompt_len
            sys_prompt_compress_ratio = sys_prompt_compress_ratio if sys_prompt_compress_ratio > 0 else 0

            # slm = torch.load("ELASTICLLM/train_slm/{}/slm_scorehead.pt".format(args.model)).cuda()
            text = decision_head_tokenizer.encode(
                SYS_PROMPT,
                return_tensors='pt'
            ).cuda()
            pred = slm(text).logits[:, :, 1].squeeze()
            pred, pred_indices = torch.sort(pred)
            # print(sys_prompt_compress_ratio)
            # pred_indices = pred_indices[int(sys_prompt_compress_ratio*sys_prompt_len):]
            pred_indices = pred_indices[:int(sys_prompt_compress_ratio*sys_prompt_len)+1]

            sys_prompt_compressed = []
            for idx in range(sys_prompt_len):
                if idx in pred_indices:
                    sys_prompt_compressed.append(sys_prompt_ids[idx])
            
            sys_prompt_compressed = decision_head_tokenizer.decode(
                sys_prompt_compressed
            )

            TEMPLATE = sys_prompt_compressed + QUERY
        elif args.mode == "Lingua2+Contextual":                
            from llmlingua import PromptCompressor

            ## Or use LLMLingua-2-small model
            llm_lingua = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True, # Whether to use llmlingua-2
            )

            sys_prompt_compressed = llm_lingua.compress_prompt(SYS_PROMPT, rate=prompt_ratio, force_tokens = ['\n', '?'])['compressed_prompt']
            TEMPLATE = sys_prompt_compressed + QUERY

            import scores
            import scores
            if args.model == "orca3b-mini":
                scores.contextual.predictor = torch.load("ELASTICLLM/Contextual_sparsity/predictors/orcamini/mlps.pt")
            elif args.model == "llama3-instruct":
                scores.contextual.predictor = torch.load("ELASTICLLM/Contextual_sparsity/predictors/llama3_instruct/mlps.pt")
            else:
                scores.contextual.predictor = torch.load("ELASTICLLM/Contextual_sparsity/predictors/{}/mlps.pt".format(args.model))
            scores.sparse.CONTEXTUAL_INFERENCE = True
            scores.contextual.ratio = model_ratio
        else:
            TEMPLATE = SYS_PROMPT + QUERY
        # print(TEMPLATE)

        prompt = TEMPLATE.format(question=dataset[i]['question'])
        # create answers
        batched_answers = []
        generated_texts = []
        for j in range(len(dataset[i]['choices']['text'])):
            answer = prompt + dataset[i]['choices']['text'][j]
            batched_answers.append(answer)
            generated_texts.append(dataset[i]['choices']['text'][j])

        inputs = tokenizer.batch_encode_plus(
            batched_answers,
            add_special_tokens=False,
            padding=True,
            return_tensors='pt'
        )
        generated_ids = tokenizer.batch_encode_plus(
            generated_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors='pt'
        )

        generated_tokens = generated_ids["input_ids"].cuda()
        generated_mask = generated_ids["attention_mask"].cuda()
        # print(inputs["input_ids"])
        pred = F.log_softmax(
            model(
                inputs["input_ids"][:, :-1].cuda(), 
                attention_mask = inputs["attention_mask"][:, :-1].cuda(),
            ).logits,
            dim=-1
        )
        max_generated_len = generated_ids["attention_mask"].shape[-1]
        pred = pred[:, -max_generated_len:, :]
        idx = generated_tokens.unsqueeze(2)
        prob = torch.gather(pred, 2, idx).squeeze(-1)
        # print(prob.shape)
        # print(generated_mask.shape)
        prob *= generated_mask

        sumprobs = torch.sum(prob, dim=1)
        sumprobs /= torch.sum(generated_mask, dim=1)
        res = torch.argmax(sumprobs.squeeze(), dim=0)
        # print(sumprobs)
        if LABELS[res] == dataset[i]['answerKey'] or LABELS_NUMBER[res] == dataset[i]['answerKey']:
            num_right += 1
print(num_right/len(dataset))

with open(args.res_save_pth, 'a') as file:
    file.write('ARC_E {} {} {} {} {}\n'.format(
        args.model,
        args.mode,
        args.prefill_SLO,
        args.decode_SLO,
        str(num_right/len(dataset))))
