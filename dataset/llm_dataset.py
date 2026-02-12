import json

import torch
from torch.utils.data import Dataset
import os

# 设置tokenizer不并行加速，避免 死锁 报错
os.environ["TOKENIZEERS_PARALLELISM"] = "false"
# 先写dataset类
class PretrainDataset(Dataset):
    def __init__(self,data_path,tokenizer,max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)
# 实现dataset内定的方法
#  自定义load_data方法
    def load_data(self,path):
        samples = []
        with open(path,"r",encoding="utf-8") as f:
            for line_num , line in enumerate(f,1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
# _len_
    def __len__(self):
        return len(self.samples)
# _getitem_
    def __getitem__(self, index):
        sample = self.samples[index]

        encoding = self.tokenizer(
            str(sample["text"]),
            max_length = self.max_length,
            padding = "max_length",
            truncation = True,
            return_tensors = "pt"
        )

        #张量形式 (max_length, )
        input_ids = encoding["input_ids"].squeeze()
        
        # 最后的形式类似：[1,1,1,0,0]
        loss_mask = input_ids != self.tokenizer.pad_token_id
        
        # 自回归,x去掉最后一个token,y去掉第一个token
        X = torch.tensor(input_ids[:-1],dtype=torch.long)
        Y = torch.tensor(input_ids[1:],dtype=torch.long)

        loss_mask = torch.tensor(loss_mask[1:],dtype=torch.long)

        return X,Y,loss_mask


import random
def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


from datasets import load_dataset
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================

        # SFT: loss_mask 用于标记需要计算loss的位置
        loss_mask = torch.tensor([l != -100 for l in labels], dtype=torch.long)

        # 自回归: X去掉最后一个token, Y去掉第一个token
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(labels[1:], dtype=torch.long)
        loss_mask = loss_mask[1:]

        return X, Y, loss_mask