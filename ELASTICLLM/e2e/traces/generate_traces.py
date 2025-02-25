# generate traces from the following datasets/slos
#      SLO      |    Dataset
# <100%, 100%>  |     OBQA
#  <80%, 90%>   |     ARC_E
#  <60%, 80%>   |    Octopus
#  <40%, 70%>   |  LlamaTouch
#  <20%, 60%>   |     PIQA
#  <20%, 50%>   |     SCIQ

# Trace details
# Prompt   |   choices    |   groundtruth  |     SLO     |   timestamp   |   dataset

import math
import numpy as np
import datasets
import pandas as pd
import os
import random

import torch

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

def skewness(alpha, sample_num):
    res = []
    sum_base = 0
    for i in range(1, 7):
        sum_base += math.exp(alpha*i)
    for i in range(1, 7):
        res.append(int(sample_num*math.exp(alpha*i)/sum_base))
    return res

def timestamp_sampling(num_per_hour, hours):
    res = []
    timenow = 0
    samples = np.random.poisson(lam=num_per_hour, size=hours)
    for sample in samples:
        for s in range(sample):
            timenow += round(1/sample, 2)
            res.append(round(timenow, 2))
    return res

arc_e_hf = datasets.load_dataset(path="allenai/ai2_arc", name="ARC-Easy", split="validation")
obqa_hf = datasets.load_dataset(path="allenai/openbookqa", name="main", split="validation")
piqa_hf = datasets.load_dataset(path="gimmaru/piqa", split="validation")
sciq_hf = datasets.load_dataset(path="allenai/sciq", split="validation")


arc_e_list = []
for r in range(len(arc_e_hf)):
    arc_e_list.append([arc_e_hf['question'][r], arc_e_hf['choices'][r]['text'], arc_e_hf['answerKey'][r]])

obqa_list = []
for r in range(len(obqa_hf)):
    obqa_list.append([obqa_hf['question_stem'][r], obqa_hf['choices'][r]['text'], obqa_hf['answerKey'][r]])

sciq_list = []
for r in range(len(sciq_hf)):
    sciq_list.append([sciq_hf['question'][r], [sciq_hf['distractor1'][r], sciq_hf['distractor2'][r], sciq_hf['distractor3'][r], sciq_hf['correct_answer'][r]], 3])

piqa_list = []
for r in range(len(piqa_hf)):
    piqa_list.append([piqa_hf['goal'][r], [piqa_hf['sol1'][r], piqa_hf['sol2'][r]], int(piqa_hf['label'][r])])

data_dir = "ELASTICLLM/Data/Octopus_refined"
task = "android_benchmark"
octopus_dev_df = pd.read_csv(os.path.join(data_dir, "dev", task + ".csv"), header=None)
octopus_val_df = pd.read_csv(os.path.join(data_dir, "val", task + ".csv"), header=None)

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U"]
octopus_choices = []
for j in range(len(LABELS)):
    octopus_choices.append(octopus_val_df.iloc[1, j+1])

octopus_val_list = []
for r in range(len(octopus_val_df)):
    octopus_val_list.append([octopus_val_df.iloc[r, 1], octopus_choices, octopus_val_df.iloc[r, -1]])

data_dir = "ELASTICLLM/Data/LlamaTouch"
task = "llamatouch_task_metadata"
llamatouch_dev_df = pd.read_csv(os.path.join(data_dir, "dev", task + ".tsv"), sep='\t', header=None)
llamatouch_val_df = pd.read_csv(os.path.join(data_dir, "test", task + ".tsv"), sep='\t', header=None)

apps = [
    'Settings',
    'Contacts',
    'Google Photos',
    'Google Play Store',
    'Files',
    'Reddit',
    'Google Maps',
    'Clock',
    'Calender',
    'Youtube',
    'Google Chrome',
    'Gmail',
    'Google Play Store, Microsoft Excel',
    'Google Play Store, Firefox',
    'Google Play Store, Nova Launcher',
    'Google Play Store, HBO Max',
    'Google Play Store, Booking',
    'Google Play Store, NewsBreak',
    'Google Play Store, Expedia',
    'Google Play Store, Duolingo',
    'Google Play Store, Facebook',
    'Google Play Store, VLC',
    'Google Play Store, eBay',
    "Google Play Store, McDonald's",
    'Google Play Store, Messages',
    'chrome',
    'Calculator',
    'Coursera',
    'English Crossword Puzzle',
    'Spotify',
    'Google Play Books',
    'X',
    'Google Podcast',
    'Google News',
    'Duolingo',
    'Google Calendar',
    'Google Keep',
    'Money Manager',
    'BBC',
    'Easy Voice Recorder',
    'Messages',
    'YouTube',
    'Camera',
    'Amazon',
    'instagram',
    'Expedia',
    'Yelp',
    'Zoom',
    'CNN',
    'Google Keep Notes',
    'Walmart',
    'Facebook',
    'DoorDash',
    'CBS Sports',
    'Amazon Shopping',
    'NewsBreak',
    'Uber',
    'Burger King',
    'Discord',
    'Amazon Prime Video',
    'Google Drive',
    'Prime TV',
    'Quora',
    'Crunchyroll',
    'evernote',
    'ESPN',
    'pinterest',
    'trello',
    'Instagram',
    'Play Books',
    'webtoon',
    'google Podcasts',
    'google maps',
    'YT Music',
    'snapchat',
    'capcut',
    'Google Tasks'
]

llamatouch_val_list = []
for r in range(len(llamatouch_val_df)):
    llamatouch_val_list.append([llamatouch_val_df.iloc[r, 1], apps, llamatouch_val_df.iloc[r, -1]])

TRACE_SIZE = 600
NUM_PER_HOUR = 5
TRACE_SKEWNESS = [0, 0.25, -0.25]

for alpha in TRACE_SKEWNESS:
    timestamps = timestamp_sampling(NUM_PER_HOUR, int(TRACE_SIZE/NUM_PER_HOUR)*2)
    nums = skewness(alpha, TRACE_SIZE)


    selected_numbers = random.sample(range(len(obqa_list)), nums[0])
    trace_slo1 = [obqa_list[_] for _ in selected_numbers]

    selected_numbers = random.sample(range(len(arc_e_list)), nums[1])
    trace_slo2 = [arc_e_list[_] for _ in selected_numbers]

    selected_numbers = random.sample(range(len(octopus_val_list)), nums[2])
    trace_slo3 = [octopus_val_list[_] for _ in selected_numbers]

    selected_numbers = random.sample(range(len(llamatouch_val_list)), nums[3])
    trace_slo4 = [llamatouch_val_list[_] for _ in selected_numbers]

    selected_numbers = random.sample(range(len(piqa_list)), nums[4])
    trace_slo5 = [piqa_list[_] for _ in selected_numbers]

    selected_numbers = random.sample(range(len(sciq_list)), nums[5])
    trace_slo6 = [sciq_list[_] for _ in selected_numbers]


    trace = []
    cnt = 0

    for i in range(len(trace_slo1)):
        trace.append([trace_slo1[i][0], trace_slo1[i][1], trace_slo1[i][2], [1.0, 1.0], cnt, 'OBQA'])
        cnt += 1

    for i in range(len(trace_slo2)):
        trace.append([trace_slo2[i][0], trace_slo2[i][1], trace_slo2[i][2], [0.8, 0.9], cnt, 'ARC_E'])
        cnt += 1
        
    for i in range(len(trace_slo3)):
        trace.append([trace_slo3[i][0], trace_slo3[i][1], trace_slo3[i][2], [0.6, 0.8], cnt, 'octopus'])
        cnt += 1

    for i in range(len(trace_slo4)):
        trace.append([trace_slo4[i][0], trace_slo4[i][1], trace_slo4[i][2], [0.4, 0.7], cnt, 'llamatouch'])
        cnt += 1

    for i in range(len(trace_slo5)):
        trace.append([trace_slo5[i][0], trace_slo5[i][1], trace_slo5[i][2], [0.2, 0.6], cnt, 'PIQA'])
        cnt += 1

    for i in range(len(trace_slo6)):
        trace.append([trace_slo6[i][0], trace_slo6[i][1], trace_slo6[i][2], [0.2, 0.5], cnt, 'SCIQ'])
        cnt += 1

    random.shuffle(trace)
    print(len(trace))
    import json
    trace_json_data = json.dumps(trace)

    with open("ELASTICLLM/e2e/traces/trace_{}.json".format(str(alpha)), 'w') as file:
        file.write(trace_json_data)
