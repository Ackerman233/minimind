from transformers import PretrainedConfig

# huggingface 的transformer类
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
 
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

#继承nn.Module类

# RMSNorm方法
class RMSNorm(nn.Module):
    
#init初始化
    def __init__(self, dim:int, eps:float=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
#norm
    def _norm(self,x):
        return x * torch.rsqrt( x.pow(2).mean(-1,keepdim=True) + self.eps )
#forward方法
    def foward(self,x):
        return self._norm(x.float()).type_as(x) * self.weight

########
#RoPE部分
#注意RoPE只是一个方法，不是一层网络；所以不需要创建类/继承
def precompt_freqs_cis(dim:int,end:int=32*1024,rope_base:float=1e6,
                       rope_scaling:Optional[dict]=None):
#写出最初的RoPE式子
    freqs = 1.0 / ( rope_base ** ( torch.arange(0,dim,2)[:dim//2].float()/dim ) )
    
    if rope_scaling is not None:
        orig_max , factor ,beta_fast , beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor",4),
            rope_scaling.get("beta_fast",4),
            rope_scaling.get("beta_slow",1)
        )
        if end / orig_max > 1.0:
        #计算corr_dim
            corr_dim = next( (i for i in range(dim//2) if 2*math.pi/freqs[i]>orig_max),dim//2)
        #计算power
            power = torch.arange(0,dim//2,device=freqs.device).float()/(max(dim//2-1,1))
        #计算beta
            beta = beta_slow + (beta_fast-beta_slow)*power
        #计算scale
            scale = torch.where(
                torch.arange(dim//2,device=freqs.device) < corr_dim ,
                (beta*factor-beta+1) / (beta+factor),
                1.0/factor
            )
        #应用scale
            freqs = freqs * scale
#生成位置索引，与频率相乘，得到完整频率矩阵
    t = torch.arange(end,device=freqs.device).float()
    freqs = torch.outer(t,freqs) #[end,dim//2]这样的矩阵
#返回一个cos和sin
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    return freqs_cos, freqs_sin

#函数：应用位置编码
def apply_rope_pos_emb(q,k,cos,sin,unsqueeze_dim=1):
    # [a,b] -> [-b,a]  旋转
    def rotate_half(x):
        # x.shape[-1] 取最后一个维度的中点
        #

        # return[后半部分,前半部分]
        return torch.cat([ -x[...,x.shape[-1]//2:] , x[...,:x.shape[-1]//2] ],
                         dim=-1)
        # 应用旋转位置编码
        # x_rot = x * cos + rotate_half(x) * sin
        # unqueeze_dim 用于在指定维度上扩展cos和sin的维度，以便与q和k进行广播操作
        q_embed = ( q * cos.unsqueeze(unsqueeze_dim) + rotate_half(q) * sin.unsqueeze(unsqueeze_dim) )
        k_embed = ( k * cos.unsqueeze(unsqueeze_dim) + rotate_half(k) * sin.unsqueeze(unsqueeze_dim) )
        return q_embed, k_embed

# 多个Q对于重复的kv
def repeat_kv(x:torch.Tensor,n_rep:int) ->torch.Tensor:
    bs, slen, num_key_value_heads, head_dim=x.shape #四维张量
    if n_rep == 1:
        return x
    
    return ( x[:,:,:,None,:]
            .expand(bs, slen, num_key_value_heads, n_rep, head_dim) #五维
            .reshape(bs, slen, num_key_value_heads * n_rep, head_dim) ) #再变回四维的，第3个维度*第4个维度


########
class Attention(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads 
            if args.num_key_value_heads is None 
            else args.num_key_value_heads
            )
        assert args.num_attention_heads % self.num_key_value_heads == 0, \
        "num_attention_heads must be divisible by self.num_key_value_heads"

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = args.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.num_key_value_heads * self.head_dim, args.hidden_size,
                                bias=False) #反过来的
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)   #残差网络
        self.dropout = args.dropout

        self.flash = hasattr(torch.nn.functional,"scaled_dot_product_attention") and args.flash_attention
    
# Forward方法
    def forward(self,x:torch.Tensor,position_embddings:Tuple[torch.Tensor,torch.Tensor],
                past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                use_cache=False,
                attention_mask:Optional[torch.Tensor]=None 
                )->torch.Tensor:
    # 投影，计算q，k，v
        bsz,seq_len,_ = x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
    # 把输入拆分成多个头，用view
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
    # q和k，使用roPE 旋转位置编码
        cos,sin=position_embddings
        xq,xk=apply_rope_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
    # 对于k和v，使用repeat(注意kv cache)
        if past_key_value is not None:
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        past_kv = (xk,xv)

        xq,xk,xv = (
            xq.transpose(1, 2), # [bsz, n_local_heads, seq_len, head_dim]
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

    # 进行attention计算，q@k^T/sqrt(d)
        if self.flash and seq_len>1 and (attention_mask is None 
                                         or torch.all(attention_mask==1)):
            attn_mask=(
                None
                if attention_mask is None
                else attention_mask.view(bsz,1,1,-1).expand(bsz,self.n_local_heads,
                                                             seq_len,-1).bool()
            )
            output = F.scaled_dot_product_attention(xq,xk,xv,attn_mask=attn_mask,
                                                    dropout_p=self.dropout if self.training else 0.0,
                                                    is_causal=True) #self.training就是训练 还是 推理模式
    # 核心部分：attn计算
        else:
            scores = (xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)
            # triu 应用掩码
            scores = scores + torch.triu(
                torch.full((seq_len,seq_len),float('-inf'),device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)
    # 最后拼接头，输出投影，返回
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask
        
            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            # 和v相乘
            output=scores@xv

    # output维度：[bsz,n_local_heads,seq_len,head_dim]
        output = output.transpose(1,2).reshape(bsz,seq_len,-1)
        output = self.resid_dropout(self.o_proj(output))
        return output,past_kv
    

########
# FFN层
class FeedForward(nn.Module):
    # 初始化
    # 升维
    # 降维
    # 门控
    # dropout
    # 激活函数
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = int(args.hidden_size*8/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)

        # 升维
        self.up_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        # 降维
        self.down_proj = nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        # 门控
        self.gate_proj = nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        # dropout
        self.dropout = nn.Dropout(args.dropout)
        # 激活函数
        self.act_fn = ACT2FN[args.hidden_act]
    
    def forward(self,x):
        return self.dropout(
            self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x) 
                )
            )


########
# Block：拼接，把GQA和FFN层拼到一起    
class MokioMindBlock(nn.Module):
    def __init__(self, layer_id:int,config:MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self,hidden_states,position_embeddings,past_key_value=None,
                use_cache=False,attention_mask=None):
        residual = hidden_states
        hidden_states, 
        present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
            )
        return hidden_states, present_key_value

######
# 组装成完整的Model(从tokenizer到RMSNorm)
class MokioMindModel(nn.Module):
    def __init__(self, config:MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers=(
            config.vocab_size,
            config.num_hidden_layers,
        )

        #把token转化成向量
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList(
            [MokioMindBlock(i,config) for i in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        #RoPE预计算
        freqs_cos, freqs_sin = precompt_freqs_cis(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        self.register_buffer("freqs_cos",freqs_cos,persistent=False)
        self.register_buffer("freqs_sin",freqs_sin,persistent=False)

    def forward(
            self,
            input_ids:Optional[torch.Tensor]=None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            **kwargs,
    ):
        batch_size,seq_len = input_ids.shape

        if hasattr(past_key_values,"layers"):
            past_key_values=None

        past_key_values = past_key_values or [None]*len(self.layers)
        
        # 计算起始位置
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 应用位置编码
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_len],
            self.freqs_sin[start_pos : start_pos + seq_len],
        )

        # 现在的缓存
        presents = []
        for layer_idx ,(layer,past_key_value) in enumerate(
            zip(self.layers,past_key_values)
        ):
            hidden_states, present=layer(
                hidden_states,
                position_embeddings,
                past_key_value = past_key_value,
                use_cache = use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        return hidden_states,presents
    
######
# CausalLM模型封顶(从Linear到Tokenizer Decoder)
class MokioMindForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config:MokioMindConfig):
        self.config = config
        super().__init__(config)

        self.model = MokioMindModel(config)

        self.lm_head=nn.Linear(
            self.config.hidden_size,
            self.config.vocab_size,
            bias=False
        )
        # 权重共享
        # 让输出层的权重和嵌入曾的权重共享
        self.model.embed_tokens.weight = self.lm_head.weight




    def forward(
            self,
            input_ids:Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            logits_to_keep:Union[int,torch.Tensor]=0,
            **args
            ):
        hidden_states, past_key_values=self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            use_cache = use_cache,
            **args,
        )
        # 如果logits to keep是整数，那就保留最后n个位置
        # 生成时只需要最后的logits来预测下一个token
        slice_indices = (
            slice(-logits_to_keep,None) 
            if isinstance(logits_to_keep,int)
            else logits_to_keep
        )
        logits =  self.lm_head(hidden_states[:,slice_indices,:])

        return CausalLMOutputWithPast(
            logits = logits,
            past_key_values = past_key_values,
            hidden_states= hidden_states 
        )