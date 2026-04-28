from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from vllm import LLM, SamplingParams

class LocalLLM:
    def __init__(
            self,
            model_path: str,
            device: str = None,
            dtype: str = "auto",
            max_new_tokens: int = 1024,
            use_vllm: bool = False,
            vllm_gpu_util: float = 0.7,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        
        if self.use_vllm:
            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                raise ImportError("vLLM error")
            
            print(f"[LocalLLM] Initializing vLLM engine with model: {model_path}...")
            

            
            self.llm = LLM(
                model=model_path,
                dtype=dtype,
                gpu_memory_utilization=vllm_gpu_util,
                trust_remote_code=True,
                enforce_eager=False,
                tensor_parallel_size=4

            )
            self.sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=0.6, top_p=0.9)
            self.tokenizer = self.llm.get_tokenizer()
            self.device = "cuda"
        
        else:

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            
            print(f"[LocalLLM] Loading HuggingFace model: {model_path} to {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            ).to(self.device)
            
            self.model.eval()
    
    def __call__(self, prompt: str) -> str:
        if self.use_vllm:
            outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text.strip()
        else:
            # --- Transformers Inference ---
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.6,
                )
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    def train(self, mode: bool = True):
        if not self.use_vllm: self.model.train(mode)
    
    def eval(self):
        if not self.use_vllm: self.model.eval()
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing only for HF model."""
        if self.use_vllm:
            print("[LocalLLM] use_vllm=True: gradient checkpointing is not applicable.")
            return
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                print("[LocalLLM] Gradient checkpointing enabled on HF model.")
            else:
                print("[LocalLLM] Underlying model does not support gradient checkpointing.")

    def compute_logprob_batch(
            self,
            prompts: list[str],
            targets: list[str],
            max_length: int = 2048,
    ):
        """
        统一接口：
        - use_vllm == True 时：使用 vLLM 的 prompt_logprobs
        - use_vllm == False 时：回退到 HF 的 calculate_log_prob_batch
        返回：Tensor [B]，每个元素是 target 部分 token logprob 之和
        """
        if not self.use_vllm:
            from .loss import calculate_log_prob_batch
            return calculate_log_prob_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                input_prompts=prompts,
                target_output=targets,
                max_length=max_length,
                device=self.device,
                requires_grad=False
            )
    
        # -------- vLLM 分支 --------
        tokenizer = self.llm.get_tokenizer()
    
        # 1. 拼 combined，并分别做 tokenization
        combined_texts = [p + t for p, t in zip(prompts, targets)]
        prompt_token_ids_list = [
            tokenizer.encode(p, add_special_tokens=False) for p in prompts
        ]
        combined_token_ids_list = [
            tokenizer.encode(ct, add_special_tokens=False) for ct in combined_texts
        ]
    
        # 2. 要求返回 prompt_logprobs（只看 prompt，不需要生成新 token）
        sampling_params = SamplingParams(
            max_tokens=1,  # 不生成新 token，只算 prompt 的 logprob
            prompt_logprobs=1  # 要求返回每个 prompt token 的 top logprobs
        )
    
        outputs = self.llm.generate(
            combined_texts,
            sampling_params,
            use_tqdm=False
        )
    
        results = []
        for prompt_ids, combined_ids, out in zip(
                prompt_token_ids_list, combined_token_ids_list, outputs
        ):
            prompt_len = len(prompt_ids)
            prompt_logprobs = out.prompt_logprobs  # 长度 ≈ len(combined_ids)
        
            # 有些实现里可能会有 off-by-one，保险起见取两者的 min
            L = min(len(prompt_logprobs), len(combined_ids))
            if prompt_len > L:
                # 极端错误情况，直接跳过 target 部分
                results.append(0.0)
                continue
        
            total_logprob = 0.0
        
            # target 对应的是 combined_ids[prompt_len : L]
            for pos in range(prompt_len, L):
                tid = combined_ids[pos]
                token_lp_dict = prompt_logprobs[pos]  # {token_id: LogProb}
            
                if tid in token_lp_dict:
                    total_logprob += token_lp_dict[tid].logprob
                else:
                    # 如果目标 token 不在 top_logprobs 里，用最小的 logprob 近似
                    total_logprob += min(x.logprob for x in token_lp_dict.values())
        
            results.append(total_logprob)
    
        return torch.tensor(results, dtype=torch.float32)

