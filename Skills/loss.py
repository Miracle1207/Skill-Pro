


import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def calculate_log_prob_batch(
        model,
        tokenizer,
        input_prompts: list[str],
        target_output: list[str],
        max_length: int,
        device,
        requires_grad: bool = True,
        ring_attn_group=None,
):
    """
    计算 batch 的 target_output 的 log-prob(sum)。
    修正版：基于 ID 拼接，确保 Prompt/Target 边界对齐准确。
    """
    
    if not input_prompts:
        return torch.tensor([], device=device)
    
    # --- 1. 安全的 Tokenization 与 拼接 (CPU) ---
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    # 分别编码，确保边界清晰
    # Prompt: 允许添加 special tokens (如 BOS)
    prompts_tokens = tokenizer(input_prompts, add_special_tokens=True, truncation=True, max_length=max_length // 2)
    # Target: 不添加 special tokens (接在 prompt 后面)，也不要自动加 BOS
    targets_tokens = tokenizer(target_output, add_special_tokens=False, truncation=True, max_length=max_length // 2)
    
    for i in range(len(input_prompts)):
        p_ids = prompts_tokens["input_ids"][i]
        t_ids = targets_tokens["input_ids"][i]
        
        # 拼接 ID
        combined_ids = p_ids + t_ids
        
        # 截断 (从后截断比较安全，但这里简单截断即可)
        if len(combined_ids) > max_length:
            combined_ids = combined_ids[:max_length]
        
        # 构建 Labels
        # Prompt 部分设为 IGNORE_INDEX
        # Target 部分保留原 ID
        # 边界由 len(p_ids) 精确控制
        p_len = len(p_ids)
        curr_labels = [IGNORE_INDEX] * p_len + combined_ids[p_len:]
        
        # 再次截断 Labels 以匹配 combined_ids 长度 (以防万一)
        curr_labels = curr_labels[:len(combined_ids)]
        
        input_ids_list.append(torch.tensor(combined_ids, dtype=torch.long))
        labels_list.append(torch.tensor(curr_labels, dtype=torch.long))
        attention_mask_list.append(torch.tensor([1] * len(combined_ids), dtype=torch.long))
    
    # Padding (Batch化)
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
    
    # --- 2. 模型前向 (GPU) ---
    target_device = model.device if hasattr(model, "device") else device
    input_ids = input_ids.to(target_device)
    attention_mask = attention_mask.to(target_device)
    labels = labels.to(target_device)
    
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # [B, L, V]
        
        # Shift logits and labels
        # Logits at t 预测 t+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # token-level CE loss
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction='none'
        )

        loss = loss.view(input_ids.size(0), -1)  # [B, L-1]

        # ====== 新增 / 替换部分 ======
        valid = (shift_labels != IGNORE_INDEX).float()
        neg_logp_tok = -loss
        logp_sum = (neg_logp_tok * valid).sum(dim=1)
        tok_cnt = valid.sum(dim=1).clamp_min(1.0)
        output_logprob = logp_sum / tok_cnt


    if requires_grad and output_logprob.grad_fn is None:
        output_logprob.requires_grad_(True)
    
    return output_logprob