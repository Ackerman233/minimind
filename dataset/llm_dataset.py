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
    
    
# ──────────────────────────────────────────────────────────────────────────────
# 3. RLAIFDataset —— 基于 AI 反馈的强化学习数据集（用于 PPO / GRPO）
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：为 RL 训练提供"问题-参考答案"对，由 actor 在线采样生成回复，
#           再由 reward model 或规则函数打分优化
# 数据格式：{"conversations": [{"content": "..."}, {"content": "..."}]}
#   - 奇数索引 (0,2,4...) 为 user 发言
#   - 偶数索引 (1,3,5...) 为 assistant 发言（最后一条为参考答案）
# 训练特点（与前三个 Dataset 的核心区别）：
#   - **不做离线 tokenize**：只返回原始字符串 prompt 和 answer，
#     让 RL trainer（PPO/GRPO）在线 rollout 时自行 tokenize，
#     因为 RL 需要动态生成回复并实时打分，无法预先固定 token 序列。
#   - create_chat_prompt 会剥离最后一条 assistant 消息，
#     将其余对话渲染为带 add_generation_prompt=True 的 prompt，
#     供 actor 模型续写；answer 保存为参考答案用于奖励计算。
#   - bos_id / eos_id 在此类中被定义但目前未用于 mask 计算，
#     保留以备后续扩展（如 reward shaping）需要。
#   - 返回值是 dict{"prompt": str, "answer": str}，而非 tensor，
#     这是 RL 数据集与 SL 数据集（返回 tensor）的最显著差异。
# ──────────────────────────────────────────────────────────────────────────────
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=jsonl_path, split="train")
        # 保留 bos_id / eos_id 以兼容未来可能的 mask 扩展
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}", add_special_tokens=False
        ).input_ids

    def __len__(self):
        return len(self.samples)
    
    def create_chat_prompt(self, conversations):
        """
        从对话列表中分离 prompt（上文）和 answer（参考答案）。

        处理逻辑：
        1. 按奇偶索引为每条消息分配 user/assistant 角色。
        2. 记录最后一条消息内容为 answer（即本轮期望的参考回答）。
        3. 用除最后一条之外的消息渲染 prompt，并开启 add_generation_prompt=True，
           使模板在末尾自动追加"assistant 开始回复"的引导标记。
        4. RL actor 收到 prompt 后进行 rollout，生成的回复与 answer 对比打分。
        """
        messages = []
        answer = ""
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
            answer = turn["content"]  # 持续更新，最终保留最后一条 assistant 内容
        # messages[:-1]：去掉最后一条 assistant 回复，只保留上下文
        # add_generation_prompt=True：在末尾追加续写引导 token，告诉模型"现在开始生成"
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer
    
    def __getitem__(self, index):
        sample = self.samples[index]
        # 返回原始字符串，不做 tokenize，由 RL trainer 在线处理
        prompt, answer = self.create_chat_prompt(sample["conversations"])

        return {"prompt": prompt, "answer": answer}
    
# ──────────────────────────────────────────────────────────────────────────────
# 4. DPODataset —— 直接偏好优化（Direct Preference Optimization）数据集
# ──────────────────────────────────────────────────────────────────────────────
# 训练目标：让模型学会"偏好好回答、远离坏回答"，使输出更符合人类偏好
# 数据格式：{"chosen": [{role, content}...], "rejected": [{role, content}...]}
#   - chosen：人类标注的更优回答对话
#   - rejected：人类标注的较差回答对话
# 训练特点：
#   - 每条样本同时返回 chosen 和 rejected 两份 tokenized 序列，
#     训练时 DPO loss 会最大化 chosen 回复的对数似然、最小化 rejected 的。
#   - loss_mask 的设计与 SFT 一致：只有 assistant 回复部分为 1，
#     其余为 0，保证对比信号仅来自模型的实际输出部分。
#   - 采用"错位"方式构造输入输出对：x 取 [:-1]，y 取 [1:]，
#     即 x[t] 预测 y[t] = input[t+1]，标准自回归格式。
#   - mask 同样错位取 [1:]，与 y 对齐，方便在训练时直接做 masked loss。
#   - max_length 默认 4096，比 SFT 更长，因为 DPO 数据通常包含完整对话上下文。
# ──────────────────────────────────────────────────────────────────────────────
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad_token_id 若不存在则回退到 0，保证补齐操作不会崩溃
        self.padding = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        # 与 SFTDataset 相同：预先 tokenize assistant 回复的起止标记，
        # 用于 generate_loss_mask 中精准定位 assistant 回复区间
        self.bos_id = tokenizer(
            f"{tokenizer.bos_token}assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f"{tokenizer.eos_token}\n", add_special_tokens=False
        ).input_ids
        self.samples = load_dataset("json", data_files=file_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample["chosen"]  # 优质回答对话列表，格式：[{role, content}, ...]
        rejected = sample["rejected"]  # 劣质回答对话列表，格式同上

        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)

        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        chosen_input_ids = chosen_encoding["input_ids"]
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding["input_ids"]
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)

        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标
        #         mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        # ！修正：返回 attention_mask，使 attention 层能屏蔽 padding token
        attention_mask_chosen = (
            torch.tensor(chosen_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()
        attention_mask_rejected = (
            torch.tensor(rejected_input_ids[:-1], dtype=torch.long) != self.padding
        ).long()

        return {
            "x_chosen": x_chosen,
            "y_chosen": y_chosen,
            "mask_chosen": mask_chosen,
            "x_rejected": x_rejected,
            "y_rejected": y_rejected,
            "mask_rejected": mask_rejected,
            "attention_mask_chosen": attention_mask_chosen,
            "attention_mask_rejected": attention_mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """
        生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回的是具体的 token id（用于 CE loss）
        - DPO 返回的是 0/1 掩码（用于 masked 对数似然计算）
        回答部分为1， 问题部分为0
        
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1，其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
