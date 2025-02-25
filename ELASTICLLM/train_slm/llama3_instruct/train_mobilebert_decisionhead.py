from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import argparse
import json
import numpy

prompt_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
model_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

prompt_ratios_dict = {1.0: 0, 0.9: 1, 0.8: 2, 0.7: 3, 0.6: 4, 0.5: 5, 0.4: 6, 0.3: 7, 0.2: 8, 0.1: 9}
model_ratios_dict = {0.2: 0, 0.3: 1, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 6, 0.9: 7, 1.0: 8}

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
        lora = torch.load("./tune_log/llama3_instruct_{pth}/adapter_model.bin".format(pth=prune_ratio))

        import scores, os
        directory = "./ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/imp.json".format(pth=prune_ratio)
        with open(directory, 'r') as file:
            imps = json.load(file)

        import scores, os
        directory = "./ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_pruned.json".format(pth=prune_ratio)
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
    directory = "./ELASTICLLM/imp/Ours/llama3_instruct/llama3_instruct_{pth}/rank_pruned.json".format(pth=prune_ratio)
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

def judge_answer(text, tokenizer, choices, label, model):
    model = model.eval()
    # print(text, choices)
    with torch.no_grad():
        # create answers
        batched_answers = []
        generated_texts = []
        for j in range(len(choices)):
            answer = text + choices[j]
            batched_answers.append(answer)
            generated_texts.append(choices[j])

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
        # print(torch.cuda.memory_reserved()/1024/1024)
        generated_tokens = generated_ids["input_ids"].cuda()
        generated_mask = generated_ids["attention_mask"].cuda()
        # print(torch.cuda.memory_reserved()/1024/1024)
        # print(inputs["input_ids"])
        b = inputs["input_ids"][:, :-1].cuda()
        c = inputs["attention_mask"][:, :-1].cuda()
        # print(torch.cuda.memory_reserved()/1024/1024)
        a = model(
            b, 
            c,
        ).logits
        # print(torch.cuda.memory_reserved()/1024/1024)
        pred = torch.nn.functional.log_softmax(
            a,
            dim=-1
        )
        # print(pred.shape)
        del a, b, c
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_reserved()/1024/1024)
        max_generated_len = generated_ids["attention_mask"].shape[-1]
        pred = pred[:, -max_generated_len:, :]
        idx = generated_tokens.unsqueeze(2)
        # print(pred.shape)
        prob = torch.gather(pred, 2, idx).squeeze(-1)
        # print(prob.shape)
        # print(generated_mask.shape)
        prob *= generated_mask
        # print(torch.cuda.memory_reserved()/1024/1024)
        sumprobs = torch.sum(prob, dim=1)
        sumprobs /= torch.sum(generated_mask, dim=1)
        res = torch.argmax(sumprobs.squeeze(), dim=0)
        if res == label:
            return True
        else:
            return False

NAME = "google/mobilebert-uncased"

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(
    NAME, ignore_mismatched_sizes=True
).cuda()

print(model)

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

prefill_slos = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decode_slos = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]



MMLU_TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies', 
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
]


MMLU_Pro_TASKS = [
    'math',
    'physics',
    'chemistery',
    'law',
    'engineering',
    'other',
    'economics',
    'health',
    'psychology',
    'business',
    'biology',
    'philosophy',
    'computer science',
    'history'
]
# [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# prompt_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# model_ratios = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# prompt_ratios_dict = {1.0: 0, 0.9: 1, 0.8: 2, 0.7: 3, 0.6: 4, 0.5: 5, 0.4: 6, 0.3: 7, 0.2: 8, 0.1: 9}
# model_ratios_dict = {0.2: 0, 0.3: 1, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 6, 0.9: 7, 1.0: 8}


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

## train decision head by score head
## for simplicity, we seperately train two heads by two model instances
for param in model.embeddings.parameters():
    param.requires_grad = False


for layer in model.encoder.layer[:12]:
    for param in layer.parameters():
        param.requires_grad = False

base_model =  AutoModelForCausalLM.from_pretrained("/data/share/Meta-Llama-3-8B", torch_dtype=torch.float32).cuda()
base_tokenizer = AutoTokenizer.from_pretrained("/data/share/Meta-Llama-3-8B")

base_model = reorder_llama3_instruct(base_model)

base_tokenizer.padding_side = "left"
base_tokenizer.pad_token = base_tokenizer.eos_token

## How it works
# sample 100 questions from MMLU and MMLU_Pro
# traverse SLO conbinations, for each conbination, traverse all model/prompt pruning ratio conbinations, choose the minimal one that correctly answers the question.
# during training, do CEL for model/prompt pruning ratio.

# step1. sample 100 questions, e.g., 1--3 questions from each subtask
mmlu = load_dataset("cais/mmlu", name ='all', split="test")
mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test").remove_columns(["question_id", "answer", "cot_content", "src"]).rename_column('options', 'choices').rename_column('category', 'subject').rename_column('answer_index', 'answer')
dataset_mmlu = []
dataset_mmlu_pro = []
for task in MMLU_TASKS:
    dataset_mmlu.append(mmlu.filter(lambda example: example['subject'] == task).select(list(range(1))))
for task in MMLU_Pro_TASKS:
    dataset_mmlu_pro.append(mmlu_pro.filter(lambda example: example['subject'] == task).select(list(range(4))))

trainset_mmlu_raw = concatenate_datasets(dataset_mmlu)
trainset_mmlu_pro_raw = concatenate_datasets(dataset_mmlu_pro)

# step2. generate SLO conbinations
SLO_conbinations = []
for prefill_slo in PREFILL_SLO:
    for decode_slo in DECODE_SLO:
        SLO_conbinations.append(prefill_slo+ " " + decode_slo)

# step3. generate inputs (samples x slos); for each input, generate outputs (model/prompt ratios)
text_inputs = []
text_scores = []
text_choices = []
labels = []

PREFIX = "You are a smart assistant that helps human sloving problems. You help them by answering questions.\n In the following part, I will give you a question with choices. Choose the correct one you think."

scorehead = torch.load("ELASTICLLM/train_slm/llama3_instruct/slm_scorehead.pt")
# for entry in tqdm(trainset_mmlu_raw):
#     question = entry['question']
#     choices = " "
#     for choice in entry['choices']:
#         choices += " " + choice
#     text_input = question + choices
#     text_inputs.append(text_input)

#     text_choices.append(entry['choices'])
#     labels.append(entry['answer'])
#     a = tokenizer.encode(text_input, return_tensors='pt', truncation=True, max_length=512).cuda()
#     text_score = scorehead(a).logits[:, :, 1].squeeze()
#     text_scores.append(text_score)
for entry in tqdm(trainset_mmlu_pro_raw):
    question = entry['question']
    choices = " "
    for choice in entry['choices']:
        choices += " " + choice
    text_input = question + choices
    text_inputs.append(text_input)

    text_choices.append(entry['choices'])
    labels.append(entry['answer'])
    a = tokenizer.encode(text_input, return_tensors='pt', truncation=True, max_length=512).cuda()
    text_score = scorehead(a).logits[:, :, 1].squeeze()
    text_scores.append(text_score)


# 向tokenizer中加入正交的special tokens
special_tokens_dict = {'additional_special_tokens': PREFILL_SLO + DECODE_SLO}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
print(model.config.hidden_size)

one_hot_matrix = torch.eye(128).cuda()
custom_embeddings = one_hot_matrix[:num_added_toks]
embeddings = model.get_input_embeddings()
new_embeddings = torch.cat([embeddings.weight, custom_embeddings], dim=0).cuda()
embeddings.weight.data = new_embeddings

model = MultiTaskBertforSeqCLS(model, num_labels_task1=len(prompt_ratios), num_labels_task2=len(model_ratios)).cuda()


inputs = []
outputs = []
for idx in tqdm(range(len(text_inputs))):
    if idx <= 10:
        continue

    if idx % 10 == 0:
        import json
        # 序列化数据
        inputs_json_data = json.dumps(inputs)
        outputs_json_data = json.dumps(outputs)
        # 写入文件
        with open("ELASTICLLM/train_slm/llama3_instruct/in_out_ckpts/inputs_{}.json".format(str(idx)), 'w') as file:
            file.write(inputs_json_data)
        with open("ELASTICLLM/train_slm/llama3_instruct/in_out_ckpts/outputs_{}.json".format(str(idx)), 'w') as file:
            file.write(outputs_json_data)

    text = text_inputs[idx]
    for prefill_slo_idx in range(len(prefill_slos)):
        for decode_slo_idx in range(len(decode_slos)):
            slo = PREFILL_SLO[prefill_slo_idx] + " " + DECODE_SLO[decode_slo_idx]
            inputs.append(slo + " " + text)
            output = [prompt_ratios[-1], model_ratios[-1]]
            for model_ratio in model_ratios:
                for prompt_ratio in prompt_ratios:
                    if model_ratio*prompt_ratio <= prefill_slos[prefill_slo_idx] and model_ratio <= decode_slos[decode_slo_idx]:
                        # print(torch.cuda.memory_reserved()/1024/1024)
                        model_elastic = get_llama3_instruct(model=base_model, ratio=model_ratio)
                        pred, pred_indices = torch.sort(text_scores[idx])
                        # print(text_scores[idx])
                        pred_indices = pred_indices[:int(prompt_ratio*len(text_scores[idx]))]
                        
                        # print(pred_indices)
                        # print(torch.cuda.memory_reserved()/1024/1024)
                        text_elastic_idx = []
                        for token_idx in range(len(text_scores[idx])):
                            if token_idx in pred_indices:
                                text_elastic_idx.append(token_idx)
                        text_ids = tokenizer.encode(text)
                        text_elastic = tokenizer.decode(
                            [text_ids[_] for _ in text_elastic_idx]
                        )

                        # print(text_choices[idx])
                        # print(labels)

                        # print(torch.cuda.memory_reserved()/1024/1024)
                        if judge_answer(model=model_elastic, choices=text_choices[idx], text=text_elastic, label=labels[idx], tokenizer=base_tokenizer):
                            output[0] = prompt_ratio
                            output[1] = model_ratio
                        # print(torch.cuda.memory_reserved()/1024/1024)
                        # print("\n")
                    continue
            outputs.append(output)


class CustomDataset(Dataset):
    def __init__(self, data, labels1, labels2):
        self.data = data
        self.labels1 = labels1
        self.labels2 = labels2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label1 = self.labels1[idx]
        label2 = self.labels2[idx]
        return data_point, label1, label2

data = []
data_labels1 = []
data_labels2 = []
for text in inputs:
    data.append(torch.tensor(
        tokenizer.encode(
            text, padding="max_length", truncation=True, max_length=512
        )
    ))


for policy in outputs:
    data_labels1.append(
        torch.tensor(prompt_ratios_dict[policy[0]])
    )
    data_labels2.append(
        torch.tensor(model_ratios_dict[policy[1]])
    )

dataset = CustomDataset(data, data_labels1, data_labels2)

# print(dataset)
# print(dataset[0])

dataloader = DataLoader(dataset, batch_size=128, num_workers=1)


criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)
model.train()
for epoch in range(1):
    pbar = tqdm(dataloader)
    for batch in pbar:
        logits1, logits2 = model(batch[0].cuda())

        # print(batch[1].cuda(), batch[2].cuda())

        loss = criterion(logits1, batch[1].cuda()) + criterion(logits2, batch[2].cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f'Training (loss: {loss.item():.4f})')
torch.save(model, "ELASTICLLM/train_slm/llama3_instruct/slm_decisionhead_llama3_instruct.pt")