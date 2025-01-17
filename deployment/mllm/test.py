# import numpy as np
# import torch

# weights = torch.load("/root/ElastiLM/deployment/mllm/tools/convertor/bin/Llama-2-7b-chat-hf-fp16.bin")

# NUM = 41968

# uint16 = format(NUM, '016b')

# uint16 = np.uint16(NUM)
# fp16 = uint16.view(np.float16)

# print(fp16)

# print(weights['model.layers.0.self_attn.k_proj.weight'][0][0].numpy())

# rr = format(
# np.uint16(
#     weights['model.layers.0.self_attn.k_proj.weight'][0][0].numpy().view('H')
# ), '016b'
# )

# print(rr)

# import torch
# import torch.nn as nn

# import transformers

# inputs_embeds = torch.randn((1, 4, 2))

# print(inputs_embeds)

# inputs_embeds = torch.cat(
#     [
#         nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
#         inputs_embeds,
#         nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
#     ],
#     dim=2,
# )

# print(inputs_embeds)

from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
import torch

# 加载预训练的 MobileBERT 分词器和模型
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased')

# 准备输入问题和上下文
text = "hello what is your name?"
inputs = tokenizer(text, return_tensors='pt')


print(model.mobilebert.embeddings.LayerNorm.weight[0])
print(model.mobilebert.embeddings.LayerNorm.bias.shape)

# print(model.mobilebert.embeddings.word_embeddings.weight[1])
# print(model.mobilebert.embeddings.position_embeddings.weight[1])
# print(model.mobilebert.embeddings.token_type_embeddings.weight[1])

# print(model.mobilebert.encoder.layer[23].bottleneck.attention.dense.weight[0])

# print(model.mobilebert.encoder.layer[23].bottleneck.attention.dense.weight[0])

# for _ in range(24):
#     print(model.mobilebert.encoder.layer[_].bottleneck.input.LayerNorm.weight[0])
#     print(model.mobilebert.encoder.layer[_].bottleneck.attention.LayerNorm.weight[0])
#     print(model.mobilebert.encoder.layer[_].attention.output.LayerNorm.weight[0])
#     print(model.mobilebert.encoder.layer[_].output.LayerNorm.weight[0])
#     for __ in range(3):
#         print(model.mobilebert.encoder.layer[_].ffn[__].output.LayerNorm.weight[0])



# print(inputs)

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)

# for k, v in model.state_dict().items():
#     print(k)