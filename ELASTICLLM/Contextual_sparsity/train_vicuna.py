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
import scores, os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model')

args = parser.parse_args()


inputs = torch.load("ELASTICLLM/Contextual_sparsity/data/{}/inputs.pt".format(args.model))
labels = torch.load("ELASTICLLM/Contextual_sparsity/data/{}/labels.pt".format(args.model))

mlps = []
optimizers = []
for i in range(32):
    mlps.append(
        torch.nn.Sequential(
            torch.nn.Linear(4096, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 11008),
            torch.nn.Sigmoid()
        ).cuda()
    )
for i in range(32):  
    optimizers.append(
        torch.optim.AdamW(
            mlps[i].parameters(),
            lr=0.0001,
            weight_decay=1e-5
        )
    )

loss_func = torch.nn.BCELoss()

for epoch in tqdm(range(10)):
    # print(epoch)
    for j in range(32):
        mlp = mlps[j]
        optm = optimizers[j]

        for i in range(5):
            # print(i)

            x = inputs[i*32+j]
            # print(x)
            x = x.view(-1, 4096).float().cuda()
            y = labels[i*32+j]
            y = (y < 0.001)
            y = y.view(-1, 11008).float().cuda()
            # print(y, y.shape)
            o = mlp(x)
            # print(x, o)
            # print(torch.all((o >= 0) & (o <= 1)))
            # print(o <= 1)
            # print(o[o > 1])
            # print(o[o < 0])
            # print(o.shape, y)
            loss = loss_func(o, y)
            optm.zero_grad()
            loss.backward()
            optm.step()
            # print("loss:", loss.item())

torch.save(f="ELASTICLLM/Contextual_sparsity/predictors/{}/mlps.pt".format(args.model), obj=mlps)