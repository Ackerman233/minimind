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
# 实现dataset内定的方法：
# 一般就是3个，load_data,_len_,_getitem_
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
        
        # 最后的形式类似：[1,1,1,0,0]，
        # 1表示这个位置是有效token，0表示这个位置是padding
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
    # 对话后处理：清理多余的<think>块
    # 特点：针对带CoT格式的模型，apply_chat_template后会在prompt中插入<think>\n\n</think>\n\n占位符，
    # 如果模型不需要CoT或者数据中没有CoT，这些占位符就没什么意义了，反而会干扰模型理解，所以我们可以随机删除一部分占位符，增加数据的多样性。
    # 保留少量空思考块（empty_think_ratio），让模型也能处理该边界情况
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
        # bos：beginning of conversation，是 assistant回答部分 开始的特殊标志
        # eos: end of conversation，是 assistant回答部分 结束的特殊标志
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()
        # 检验有没有function_calling的子块，有的话就传入prompt
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # 让所有input_id都为-100
        # 因为-100是交叉熵损失函数默认忽视的值，就是-100不计算的部分
        labels = [-100] * len(input_ids)
        i = 0
        # 遍历整个input_ids，滑动窗口
        # 目标：把问题的部分全部忽略，只留回答
        while i < len(input_ids):
            # 找到bos
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    # 找到eos
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                    
                # 恢复到本身，让回答部分变成-100的labels变回去input_ids的部分
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 是否随机插入system prompt
        conversations = pre_processing_chat(sample['conversations'])
        # 把对话转化成文本
        prompt = self.create_chat_prompt(conversations)
        # 清理空think块
        prompt = post_processing_chat(prompt)
        
        # tokenizer截断，补足pad
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 生成标签，只让assitant参与loss计算
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