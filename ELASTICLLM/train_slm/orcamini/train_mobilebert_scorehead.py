from transformers import AutoModelForTokenClassification, AutoTokenizer, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

NAME = "google/mobilebert-uncased"

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForTokenClassification.from_pretrained(
    NAME, num_labels=2, ignore_mismatched_sizes=True
).cuda()

print(model)

## train score head first
## for simplicity, we seperately train two heads by two model instances
for layer in model.mobilebert.encoder.layer[:12]:
    for param in layer.parameters():
        param.requires_grad = False

meeting_bank_comp = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        label = self.labels[idx]
        return data_point, label

data = []
labels = []
for sample in tqdm(meeting_bank_comp):
    assert len(sample["prompt_list"]) == len(sample["compressed_prompt_list"])
    for i in range(len(sample["prompt_list"])):
        ori = tokenizer.encode(sample["prompt_list"][i], padding="max_length", truncation=True, max_length=512)
        compre = tokenizer.encode(sample["compressed_prompt_list"][i], padding="max_length", truncation=True, max_length=512)
        label = []
        pointer1=0
        pointer2=0
        while pointer1 < len(ori):
            if ori[pointer1] == compre[pointer2]:
                label.append(1)
                pointer1 += 1
                pointer2 += 1
            else:
                label.append(0)
                pointer1 += 1
        data.append(torch.tensor(ori))
        labels.append(torch.tensor(label))

dataset = CustomDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2,)

optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):
    pbar = tqdm(dataloader)
    for batch in pbar:
        outputs = model(batch[0].cuda(), labels=batch[1].cuda())
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f'Training (loss: {loss.item():.4f})')
torch.save(model, "ELASTICLLM/train_slm/orcamini/slm_scorehead.pt")