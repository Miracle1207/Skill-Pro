import torch
import numpy as np
import json

from typing import Callable, Dict, Any, List, Tuple
from collections import defaultdict
from data_structures import Skill
from .loss import calculate_log_prob_batch

import re

import math

import random
from typing import List

def build_success_failure_eval_batch(
    buffer: List,
    batch_size: int,
) -> List:
    """
    Build an evaluation batch using success + failure trajectories only.

    Args:
        buffer: List of experience objects, each with a `.reward` attribute.
        batch_size: Total number of trajectories to return.

    Returns:
        eval_exps: List of selected experiences (shuffled).
    """
    if not buffer:
        return []

    if len(buffer) <= batch_size:
        eval_exps = list(buffer)
        random.shuffle(eval_exps)
        return eval_exps

    half = batch_size // 2
    if half == 0:
        # batch_size == 1 的极端情况，直接取 reward 最大的
        best = max(buffer, key=lambda e: e.reward)
        return [best]

    # 按 reward 排序（升序）
    sorted_exps = sorted(buffer, key=lambda e: e.reward)

    # 失败修复：最低 reward
    failure_exps = sorted_exps[:half]

    # 成功提炼：最高 reward
    success_exps = sorted_exps[-half:]

    eval_exps = failure_exps + success_exps
    random.shuffle(eval_exps)
    return eval_exps


def get_priority(
    sk,
    pool_skills,
    t: int,
    n_min: int = 15,
    elite_gap: float = 0.01,       # gap<=elite_gap 认为已接近顶级，稳住不演化
    freq_scale: float = 30.0,      # 可靠性饱和速度（用于 confidence）
    impact_scale: float = 50.0,    # 影响力饱和速度（用于 impact，替代 sqrt(freq) 防垄断）
    age_scale: float = 8.0,        # 老化惩罚尺度（调小更能压住“老且高频”的垄断）
    cooldown_iters: int = 3,       # 冷却期：刚演化过的 skill 暂时不再演化
):
    """
    Priority for choosing which skill to evolve (higher = evolve earlier).

    Fixes for "always evolve the highest-freq skill":
    - Add a short cooldown window after each evolution.
    - Replace impact = sqrt(freq) with a saturating impact to avoid frequency monopoly.
    - Use a slightly stronger aging penalty (smaller age_scale by default).

    Assumptions:
    - sk has attributes: frequency, avg_gain, maturity
    - sk optionally has: last_evolved_iter (int). If missing, treated as never evolved.
    """

    # ---------- 0) 新生代：证据不足，不演化 ----------
    freq = max(int(getattr(sk, "frequency", 0)), 0)
    if freq < n_min:
        return 0.0

    # ---------- 0.5) 冷却期：刚演化过就先别再演化 ----------
    last_evolved = getattr(sk, "last_evolved_iter", None)
    if last_evolved is not None:
        try:
            if int(t) - int(last_evolved) < cooldown_iters:
                return -0.01
        except Exception:
            # if parsing fails, ignore cooldown
            pass

    # ---------- 1) 参考顶级水平：成熟 skill 中最高 avg_gain ----------
    mature_sks = [o for o in pool_skills if max(int(getattr(o, "frequency", 0)), 0) >= n_min]
    if not mature_sks:
        return 0.0
    ref = max(float(getattr(o, "avg_gain", 0.0)) for o in mature_sks)

    # ---------- 2) 提升空间：离顶级越远越该演化 ----------
    avg_gain = float(getattr(sk, "avg_gain", 0.0))
    gap = max(0.0, ref - avg_gain)

    # ---------- 3) 精英保护：接近顶级就稳住 ----------
    if gap <= elite_gap:
        return 0.0

    # ---------- 4) 影响力：用得多更值得优化（饱和，防垄断） ----------
    impact = 1.0 - math.exp(-freq / max(impact_scale, 1e-6))   # in (0,1)

    # ---------- 5) 可靠性：freq 越大越可信（饱和） ----------
    confidence = 1.0 - math.exp(-freq / max(freq_scale, 1e-6))  # in (0,1)

    # ---------- 6) 老化惩罚（soft）：越老越谨慎，但不归零 ----------
    maturity = max(float(getattr(sk, "maturity", 0.0)), 0.0)
    age_factor = math.exp(-maturity / max(age_scale, 1e-6))

    return impact * confidence * gap * age_factor



class SkillEvolution:
    def __init__(self, llm_agent: Any, threshold: int = 6, epsilon: float = 0.2):
        """
        负责 Skill 的演化逻辑调度。
        threshold: 触发演化所需的轨迹积累数量。
        epsilon: PPO 截断系数 (例如 0.2)。
        """
        self.evolver_model = llm_agent
        self.threshold = threshold
        self.epsilon = epsilon
        # 按 Skill Name 存储积累的经验轨迹
        self.experience_buffer = defaultdict(list)


    def _tag(self, text: str, tag: str):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.I | re.S)
        return m.group(1).strip() if m else None


    def _parse_json_safely(self, text: str):
        if not text:
            return None
    
        # 1) extract fenced content if present
        m = re.search(r"```json\b(.*?)```", text, flags=re.S | re.I)
        s = m.group(1) if m else text
    
        # 2) grab the outermost JSON object (greedy to include nested braces)
        m = re.search(r"\{.*\}", s, flags=re.S)
        if not m:
            return None
        j = m.group(0)
    
        # 3) minimal fixes for common Python-ish booleans/null
        j = re.sub(r"\bTrue\b", "true", j)
        j = re.sub(r"\bFalse\b", "false", j)
        j = re.sub(r"\bNone\b", "null", j)
    
        # 4) parse
        try:
            return json.loads(j)
        except json.JSONDecodeError:
            return None


    def _generate_semantic_gradient(self, old_skill, experience: Any) -> Tuple[Dict[str, str], bool]:
        """
        Gemma-2-9b 优化版：基于诊断的结构化语义梯度生成
        Math: g_i = (g^(I), g^(π), g^(β))
        """
    
        # 1. 准备 Skill 描述 (假设 format_for_llm 返回清晰的定义)
        skill_info = old_skill.format_for_llm()
    
        # 2. 构造 Prompt
        # 技巧：使用清晰的分隔符，减少学术词汇，使用直白的指令
        prompt = f"""### Role
    You are a Skill Doctor. You optimize an agent's skill based on execution history.

    ### Context
    **Skill Definition:**
    {skill_info}

    **Execution Trace:**
    {experience.trajectory}

    **Result (Reward):**
    {experience.reward}

    ### Task
    1. **Diagnosis**: Identify the ROOT CAUSE of the outcome (Success or Failure).
    2. **Prescription**: Map the cause to one or more components to generate updates:
       - **Initiation (Start)**: Should it have started here? Or did it start when it shouldn't?
       - **Policy (Step-by-step)**: Did it hallucinate? Miss a step? Make a wrong move?
       - **Termination (Stop)**: Did it stop too early (incomplete)? Or loop forever?

    ### Constraints
    - If a component is fine, keep it empty string "".
    - "is_related": Set to False only if the skill was NOT used or completely irrelevant.
    - "semantic_gradient": Write concrete, actionable instructions (e.g., "Add a check for X", "Don't stop until Y").

    ### Output format (JSON ONLY)
    ```json
    {{
        "diagnosis": "Brief explanation of why it failed/succeeded...",
        "is_related": true,
        "semantic_gradient": {{
            "initiation": "...",
            "policy": "...",
            "termination": "..."
        }}
    }}
    ```"""
    
        # 3. 模型推理
        raw = self.evolver_model(prompt).strip()
    
        # 4. 安全解析
        data = self._parse_json_safely(raw)
    
        if not isinstance(data, dict):
            return {"initiation": "", "policy": "", "termination": ""}, False
    
        is_related = bool(data.get("is_related", False))
    
        # 提取梯度，如果没有则默认为空
        grads = data.get("semantic_gradient", {})
        final_grads = {
            "initiation": str(grads.get("initiation", "")).strip(),
            # 注意：这里将 math中的 pi 映射为代码中的 policy/execution，保持一致性即可
            "execution": str(grads.get("policy", "") or grads.get("execution", "")).strip(),
            "termination": str(grads.get("termination", "")).strip()
        }
    
        # 清理无效输出 (常见于 9B 模型偶尔输出 "None" 或 "N/A")
        for k, v in final_grads.items():
            if v.lower() in ["none", "n/a", "null", "no change", "keep as is"]:
                final_grads[k] = ""
    
        return final_grads, is_related


    def evolve(self, old_skill: "Skill", sampled_experience: List[Any], threshold: float = 0.5) -> Tuple[
        "Skill", str]:
        """
        梯度上升进化：ω' = ω ⊕ Aggregate({g_i})
        """
    
        # --- 1) 收集个体语义梯度 g_i ---
        gradients_list = []
        related_count = 0
    
        for exp in sampled_experience:
            g_i, is_related = self._generate_semantic_gradient(old_skill, exp)
            if is_related:
                related_count += 1
                # 我们只聚合与该 Skill 相关的梯度信号
                gradients_list.append(g_i)
    
        valid_count = len(sampled_experience)
        relevance_ratio = (related_count / valid_count) if valid_count > 0 else 0.0
    
        # 判定是进行梯度更新 (REFINE) 还是由于表现太差需要重起炉灶 (NEW)
        evolution_type = "REFINE" if relevance_ratio >= threshold else "NEW"
    
        # --- 2) 执行聚合与更新 (LLM-driven Aggregate & Update) ---
        # 我们将两个数学步骤合并为一个高效的 LLM 调用，减少推理损耗
        new_skill_data = self._run_semantic_update(old_skill, gradients_list, evolution_type)
        if not new_skill_data:
            return None, f"{evolution_type}_FAILED"
        # --- 3) 构建新 Skill 实例 ---
        suggested_name = new_skill_data.get("skill_name", "RefinedSkill")
    
        if evolution_type == "REFINE":
            new_version = old_skill.version + 1
            base_name = re.sub(r"_v\d+$", "", old_skill.name)
            new_name = f"{base_name}_v{new_version}"
            parent_id = old_skill.name
        else:
            new_name = suggested_name
            new_version = 1
            parent_id = None
    
        new_sk = Skill(
            name=new_name,
            initiation=new_skill_data.get("initiation", ""),
            policy=new_skill_data.get("policy", []),
            termination=new_skill_data.get("termination", ""),
            parent_id=parent_id,
            version=new_version
        )
    
        return new_sk, evolution_type

    def _run_semantic_update(self, old_skill: "Skill", g_list: List[Dict], evolution_type: str) -> Dict:
        """
        执行 ω' = ω ⊕ Aggregate({g_i})
        针对 9B 模型优化的极简 Prompt
        """
    
        # 格式化梯度列表，供聚合使用
        formatted_gs = ""
        for i, g in enumerate(g_list):
            formatted_gs += f"Gradient {i + 1}:\n- initiation: {g['initiation']}\n- execution: {g['execution']}\n- termination: {g['termination']}\n\n"
    
        if not g_list:
            formatted_gs = "No specific gradients. Perform a general improvement based on success/failure patterns."
    
        if evolution_type == "REFINE":
            mode_instruction = f"""[MODE: GRADIENT ASCENT]
    Apply the aggregated gradient to refine the Skill components:
    - initiation: Refine trigger to prevent failure.
   - policy: Update 3-5 steps to fix decision logic.
   - termination: Update the "Stop IF" condition to strictly verify the outcome.
    Keep the core intent of: {old_skill.name}."""
        else:
            mode_instruction = """[MODE: NEW DISCOVERY]
    The current Skill is irrelevant or fundamentally flawed.
    Synthesize a NEW Skill structure based on the gradients."""
    
        prompt = f"""### Role
    You are a Skill Evolver. Your goal is to apply a semantic gradient update to a Skill.

    ### Context
    **Original Skill (ω):**
    {old_skill.format_for_llm()}

    **Semantic Gradients (g_i):**
    {formatted_gs}

### Task 1: Batch-level Aggregation (g_bar)
Identify the **systematic weaknesses** across all gradients. Filter out noise and trajectory-specific details. Focus only on recurring failure patterns.

### Task 2: Semantic Update (ω' = ω ⊕ g_bar)
Apply the aggregated gradient to refine the Skill components:
- **Initiation (I)**: Refine the "IF" condition to ensure the skill only starts in valid, high-success states.
- **Policy (π)**: Update the 3-5 reasoning steps to bypass the identified failure patterns.
- **Termination (β)**: Update the "Stop IF" condition to strictly verify the outcome.

### Constraints
- MODE: {"REFINE existing logic" if evolution_type == "REFINE" else "CREATE new logic"}.
- Return ONLY a valid JSON object.

    ### Output format
    ```json
    {{
      "skill_name": "Concise_Name",
      "initiation": "IF... AND...",
      "policy": ["Step 1...", "Step 2...", "Step 3..."],
      "termination": "Stop IF..."
    }}
    ```
    Please output the JSON now: """
        for _ in range(5):
            raw = self.evolver_model(prompt)
            if raw and raw.strip():
                # 只要拿到非空字符串，解析并返回（若解析失败 or {} 会返回空字典）
                return self._parse_json_safely(raw) or {}
    
            # 5 次全部失败，返回空字典
        return {}

    def run_skill_evolution_with_verification(
            self,
            skill_pool: Any,
            new_experiences: List[Any],
            build_prompt_fn: Callable,
            baselines: Dict[str, float],
            acceptance_margin: float = 0.001,
            max_evolutions_per_step: int = 2,
            # max_evolutions_per_step: int = 1,
            best_of_n: int = 3,  # 引入 Best-of-N 采样
            current_iteration: int = 0,
            ablation_type: str = "none"
    ) -> List[Dict]:
        """
        支持 Best-of-N 采样与 J-Score 验证的 Skill 演化调度函数
        """
        evolution_logs = []
    
        # 1. 经验分发
        for exp in new_experiences:
            if exp.skill and exp.skill != "None":
                skill_names = exp.skill.split(";")
                for sk_name in skill_names:
                    self.experience_buffer[sk_name].append(exp)
    
        all_skills = list(skill_pool.get_all())
        ready_candidates = []
    
        for sk in all_skills:
            buffered_exps = self.experience_buffer.get(sk.name, [])
            if len(buffered_exps) >= self.threshold:
                ready_candidates.append(sk)
    
        if not ready_candidates:
            return []
    
        ready_candidates.sort(key=lambda o: get_priority(o, pool_skills=all_skills, t=current_iteration), reverse=True)
        selected_parents = ready_candidates[:max_evolutions_per_step]
    
        # 3. 对选中的 Parent 执行 Best-of-N 演化
        for old_sk in selected_parents:
            print(f"🧬 [Best-of-{best_of_n} Evolution] Selecting '{old_sk.name}' "
                  f"(Freq: {old_sk.frequency}, Gain: {old_sk.avg_gain:.3f})")
        
            # 获取用于评估的数据集
            # eval_exps = self.experience_buffer[old_sk.name][-self.threshold:]
            buffer = self.experience_buffer.get(old_sk.name, [])
            eval_exps = build_success_failure_eval_batch(buffer, self.threshold)

            # 预处理数据：展平 transitions 以进行批量 logprob 计算
            all_states = []
            all_actions = []
            all_envs = []
            traj_boundaries = [0]
            for exp in eval_exps:
                for trans in exp.transitions:
                    all_states.append(trans['state'])
                    all_actions.append(trans['action'])
                    all_envs.append(exp.env_name)
                traj_boundaries.append(len(all_states))
        
            if not all_states:
                continue
        
            old_log_probs = self._compute_skill_logprobs(old_sk, all_states, all_actions, all_envs, build_prompt_fn)
        
            best_cand_sk = None
            best_cand_j = -float('inf')
            best_cand_type = "REFINE"
        
            # --- Best-of-N 采样开始 ---
            print(f"  Sampling {best_of_n} candidates...")
            for n_idx in range(best_of_n):

                if ablation_type == "wo_sg":
                    cand_sk, cand_type = self.evolve_without_gradient(old_sk, eval_exps)
                else:
                    # 执行标准实验：带梯度
                    cand_sk, cand_type = self.evolve(old_sk, eval_exps)
                if cand_sk is None or getattr(cand_sk, "name", None) == "None":
                    continue
            
                # 计算候选者的 LogProbs
                new_log_probs = self._compute_skill_logprobs(cand_sk, all_states, all_actions, all_envs,
                                                              build_prompt_fn)

                if (not np.isfinite(new_log_probs).all()) or (not np.isfinite(old_log_probs).all()):
                    current_j = -1e9
                else:
                    total_j_score = 0.0
                    total_steps = 0
    
                    for i, exp in enumerate(eval_exps):
                        task_baseline = baselines.get(exp.env_name, 0.0)
                        start, end = traj_boundaries[i], traj_boundaries[i + 1]
                        T = end - start
                        if T <= 0:
                            continue
        
                        step_log_ratio = np.clip(new_log_probs[start:end] - old_log_probs[start:end], -10, 10)
                        step_ratio = np.exp(step_log_ratio)
        
                        traj_adv = float(exp.reward - task_baseline)
                        step_adv = traj_adv / float(T)
        
                        step_surr1 = step_ratio * step_adv
                        step_surr2 = np.clip(step_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * step_adv
                        step_obj = np.minimum(step_surr1, step_surr2)
        
                        total_j_score += float(step_obj.sum())
                        total_steps += T
    
                    current_j = total_j_score / max(total_steps, 1)

                print(f"    Candidate {n_idx + 1}: J={current_j:.4f}")
            
                # 记录表现最好的候选者
                if current_j > best_cand_j:
                    best_cand_j = current_j
                    best_cand_sk = cand_sk
                    best_cand_type = cand_type

            if best_cand_sk is None:
                accepted = False
            elif ablation_type == "wo_ppo":
                accepted = True  # without PPO Gate, always accept the best candidate
            else:
                accepted = best_cand_j > acceptance_margin
            
            log_item = {
                "parent": old_sk.name,
                "candidate": getattr(best_cand_sk, "name", None),
                "type": best_cand_type,
                "j_score": float(best_cand_j),
                "action": "NONE",
                "sample_count": len(eval_exps),
                "best_of_n": best_of_n
            }
        
            if accepted:
                old_sk.last_evolved_iter = current_iteration
                best_cand_sk.maturity = 0
                skill_pool.add_skill(best_cand_sk)
                action = "REFINE_CANDIDATE_ADDED" if best_cand_type == "REFINE" else "NEW_OPTION_ADDED"
                log_item['action'] = action
                self.experience_buffer[old_sk.name] = []  # 成功后清空 buffer
                print(f" ✅ [Accepted] {action} (Best J={best_cand_j:.4f})")
            else:
                # 未通过验证，保留一半数据继续积累，注意这里要重新从 buffer 取出完整数据
                current_buffer = self.experience_buffer.get(old_sk.name, [])
                self.experience_buffer[old_sk.name] = current_buffer[-(self.threshold // 2):]
                print(f" ❌ [Rejected] All candidates failed. Best J={best_cand_j:.4f} below margin.")
        
            evolution_logs.append(log_item)
    
        return evolution_logs
    
    
    def _compute_skill_logprobs(self, skill, states, actions, envs, build_prompt_fn):
        sk_text = skill if isinstance(skill, str) else skill.format_for_llm()
        prompts = [build_prompt_fn(s, sk_text, env) for s, env in zip(states, envs)]
        
        if hasattr(self.evolver_model, "use_vllm") and self.evolver_model.use_vllm:
            log_probs = self.evolver_model.compute_logprob_batch(prompts, actions)
        else:
            with torch.no_grad():
                log_probs = calculate_log_prob_batch(
                    model=self.evolver_model.model,
                    tokenizer=self.evolver_model.tokenizer,
                    input_prompts=prompts,
                    target_output=actions,
                    max_length=2048,
                    device=self.evolver_model.device,
                    requires_grad=False
                )
            
        return log_probs.cpu().numpy()

    def _generate_neutral_summary(self, old_skill, experience: Any) -> Tuple[str, bool]:
        """
        LLM-based neutral summary:
          - is_related: same meaning as your semantic-gradient version
          - summary: facts-only compression of trajectory (no advice, no diagnosis)
        """

        traj = str(getattr(experience, "trajectory", ""))
        # hard truncate to avoid immediate context overflow; keep it minimal & fair

        prompt = f"""You compress trajectories.

[TRAJECTORY]
{traj}

[REWARD]
{experience.reward}

Task:
Write a neutral, factual summary of observable events in the trajectory.

Rules:
- Facts only. Describe observable events.
- No diagnosis, no reasoning, no explanations.
- No advice or suggestions.

[OUTPUT — EXACT FORMAT]
Return ONLY one JSON object wrapped in triple backticks:
```json
{{
  "summary": "..."
}}
    ```"""

        raw = self.evolver_model(prompt)
        data = self._parse_json_safely(raw)
        summary = (data.get("summary", "") if isinstance(data, dict) else "").strip()
        return summary

    def evolve_without_gradient(
            self,
            old_skill: "Skill",
            sampled_experience: List[Any],
    ) -> Tuple["Skill", str]:
        """
        Ablation: w/o semantic gradient.
        Use neutral summaries (facts-only trajectory compression) to generate candidates.
        No causal attribution, no related/unrelated split.
        """
    
        summaries = []
    
        # 1) Collect neutral summaries (facts only)
        for exp in sampled_experience:
            summary = self._generate_neutral_summary(old_skill, exp)
            if not summary:
                continue
            summaries.append(f"- (Reward {exp.reward}): {summary}")
    
        aggregated = "\n".join(summaries) if summaries else "- (No valid summaries.)"
        evolution_type = "REFINE"
    
        parent_block = f"""
    Refine the PARENT skill. Preserve structure and intent.

    [PARENT OPTION]
    {old_skill.format_for_llm()}
    """
    
        # 3) Prompt: same structure, just evidence source changed
        prompt = f"""You are an Evolution Operator.

    Goal: Output ONE refined skill based on the provided evidence.

    {parent_block}

    [TRAJECTORY SUMMARIES]
    {aggregated}

    [OUTPUT FORMAT — JSON ONLY]
JSON must be valid: no trailing commas. Use only state/history checkable terms; avoid ‘successfully submitted’ unless feedback is available.

Output EXACTLY ONE skill in STRICT JSON.
NO explanations. NO comments. NO extra text.

{{
  "skill_name": "Use a short, concrete SkillName that best describes when this skill should be applied and what decision behavior it enforces",
  "target_situation": "IF ... AND ... (fully checkable, preventive conditions)",
  "recommended_strategy": [
    "S1: ...",
    "S2: ...",
    "S3: ...",
    "S4: ..."
  ],
  "termination_condition": "Stop IF ... (fully checkable, constraint-satisfying condition)"
}}
    """
    
        raw_output = self.evolver_model(prompt)
        data = self._parse_json_safely(raw_output)
        if not isinstance(data, dict):
            data = {}
    
        suggested_name = data.get("skill_name", "None")
    
        # 4) Naming / versioning: always refine parent
        new_version = old_skill.version + 1
        base_name = re.sub(r"_v\d+$", "", old_skill.name)
        new_name = f"{base_name}_v{new_version}"
        parent_id = old_skill.name
    
        new_sk = Skill(
            name=new_name,
            initiation=data.get("target_situation", ""),
            policy=data.get("recommended_strategy", []),
            termination=data.get("termination_condition", ""),
            parent_id=parent_id,
            version=new_version
        )
        return new_sk, evolution_type
