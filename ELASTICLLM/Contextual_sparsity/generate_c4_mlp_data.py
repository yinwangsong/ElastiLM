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

import sys

import scores

import numpy as np
import random
import argparse

# 获取当前文件的路径
current_path = os.path.dirname(__file__)

# 获取两级父目录的路径
parent_path = os.path.abspath(os.path.join(current_path, '../../LLMPruner/'))

# 将父目录路径添加到sys.path中
if parent_path not in sys.path:
    sys.path.append(parent_path)


parser = argparse.ArgumentParser()
parser.add_argument('--model')

args = parser.parse_args()

if args.model == "llama":
    model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
if args.model == "vicuna":
    model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
if args.model == "llama3":
    model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")
if args.model == "llama3_instruct":
    model = AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B-Instruct")
if args.model == "orcamini":
    model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b", torch_dtype=torch.float16).cuda()
    tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")


from LLMPruner.datasets.example_samples import get_examples

text1 = get_examples('c4', tokenizer, 500, seq_len = 128).cuda()


for i in range(5):
    import scores
    scores.sparse.CONTEXTUAL_TRAINING = True
    model.eval()
    text = text1[i*100: (i+1)*100]
    with torch.no_grad():
        model(text)


torch.save(f="ELASTICLLM/Contextual_sparsity/data/{}/inputs.pt".format(args.model), obj=torch.stack(scores.contextual.inputs, dim=0))
torch.save(f="ELASTICLLM/Contextual_sparsity/data/{}/labels.pt".format(args.model), obj=torch.stack(scores.contextual.labels, dim=0))