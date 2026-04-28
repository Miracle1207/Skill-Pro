from typing import Any, List, Dict
import numpy as np
import json
import time
import re
from collections import Counter
import swanlab
import os
try:
    from alfworld.agents.environment import get_environment
    import alfworld.agents.modules.generic as generic
    HAS_ALFWORLD = True
except ImportError:
    HAS_ALFWORLD = False

from pool_managers import ExperiencePool, GoldenExperiencePool
from Skills.skill_pool import SkillPool
from Skills.skill_evolution import SkillEvolution
import textarena as ta
from utils.utils import set_seed, is_local_model, build_log_file_path
from utils.local_llm import LocalLLM
from data_structures import Experience
import yaml


class SkillMDP:
    def __init__(self, args):
        self.args = args
        if getattr(args, "load_pool_path", None) and args.test == True:
        # if getattr(args, "load_pool_path", None):
            it = getattr(args, "load_iteration", -1)
            self.skill_pool = SkillPool.from_json(args.load_pool_path, iteration=it, max_size=args.pool_size)
            print(f"load skill pool: {args.load_pool_path}, iteration: {it}")
        else:
            self.skill_pool = SkillPool(max_size=args.pool_size)
        self.experience_pool = ExperiencePool(max_size=int(1e6))
        self.golden_pool = GoldenExperiencePool(max_size=20)
        
        self.env_list = args.env_names.split(",") if hasattr(args, "env_names") else [args.env_name]
        self.task_baselines = {name: 0.0 for name in self.env_list}
        self.global_baseline = 0.0  # 用于日志的总基准
        self.ema_alpha = 0.1
        # self.warm_up_flag = True
        self.warm_up_flag = False
        self.record_tokens = True
        
        if any("alfworld" in name.lower() for name in self.env_list):
            config_path = "configs/base_config.yaml"  # 确保这个文件在你运行程序的目录下
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"[ALFWorld] Successfully loaded config from {config_path}")
            else:
                print(f"[Warning] {config_path} not found. Using minimal fallback dictionary.")
                config = {
                    'env': {'type': 'AlfredTWEnv', 'max_nb_steps': 50},
                    'dataset': {'data_path': os.path.expanduser('~/.cache/alfworld')},
                    'logic': {'get_admissible_commands': True}
                }
            env_type = config['env']['type']
            if args.test == True:
                type = "eval_out_of_distribution"
            else:
                type = "train"
            raw_env = get_environment(env_type)(config, train_eval=type)
            self.alfworld_env = raw_env.init_env(batch_size=1)
        
        if is_local_model(args.agent_name):
            print(f"[LLM] Loading local policy model: {args.agent_name}")
            self.llm_policy = LocalLLM(
                model_path=args.agent_name,
                use_vllm=True,
                vllm_gpu_util=0.75
            )
        else:
            print(f"[LLM] Using OpenRouter remote model: {args.agent_name}")
            self.llm_policy = ta.agents.OpenRouterAgent(model_name=args.agent_name)
        
        if args.agent_name == args.ge_model_name:
            self.skill_evolver = SkillEvolution(llm_agent=self.llm_policy)
        else:
            evolver_llm = LocalLLM(model_path=args.ge_model_name) if is_local_model(args.ge_model_name) else ta.agents.OpenRouterAgent(model_name=args.ge_model_name)
            self.skill_evolver = SkillEvolution(llm_agent=evolver_llm)
        
        exp_name, self.log_file_path = build_log_file_path(args)
        swanlab.init(
            project="AgentEvolution",
            experiment_name=exp_name,
            config=vars(args)  # 将所有参数保存到配置中
        )
    
    def run_skill_mdp(self):
        set_seed(self.args.seed)
        training_logs = []
        for it in range(self.args.max_iters):
            if self.args.test == False:
                eps_decay_iters = 100
                progress = min(it / eps_decay_iters, 1.0)
                curr_eps = self.args.epsilon_initial + progress * (0.05 - self.args.epsilon_initial)
                mode_str = "OPTIMIZATION MODE"
            else:
                curr_eps = 0.3
                mode_str = "EPISODE RUN MODE"
                
            print(f"\n================ Iteration {it + 1}/{self.args.max_iters} ({mode_str} Epsilon: {curr_eps:.3f}) ================")
            if it >= 5 and self.args.test == False and self.warm_up_flag:
                self.warm_up_flag = False
            
            all_iter_experiences = []
            iter_task_returns = {}
            iter_skill_usage_logs = {}  # 新增：用于日志记录
            
            for env_name in self.env_list:
                print(f" > Task: {env_name}")
                task_avg, task_exps, usage_logs = self.run_episodes_in_env(env_name, curr_eps)
                
                all_iter_experiences.extend(task_exps)
                iter_task_returns[env_name] = task_avg
                iter_skill_usage_logs[env_name] = usage_logs  # 保存到临时变量
                
                if self.task_baselines[env_name] == 0:
                    self.task_baselines[env_name] = task_avg
                else:
                    self.task_baselines[env_name] = (1 - self.ema_alpha) * self.task_baselines[
                        env_name] + self.ema_alpha * task_avg
                print(
                    f"🌟 Env:{env_name} | Avg episodes return: {task_avg:.4f} | Updated Global Baseline: {self.task_baselines[env_name]:.4f}")
            
            # --- Bug 修复 2: 同步更新 global_baseline 用于日志记录 ---
            self.global_baseline = sum(self.task_baselines.values()) / len(self.task_baselines)
            
            self.skill_pool.update_pool_status()
            
            if self.args.test == False and self.args.MDP_type == "SMDP":
                print("🧬 Running Skill Evolution and Verification...")
                evolution_logs = self.skill_evolver.run_skill_evolution_with_verification(
                    skill_pool=self.skill_pool,
                    new_experiences=all_iter_experiences,
                    build_prompt_fn=self.build_decision_prompt,
                    baselines=self.task_baselines,
                    acceptance_margin=self.args.acceptance_margin,
                    current_iteration=it + 1,
                    ablation_type=self.args.ablation_type,
                )
                if self.args.ablation_type == "wo_score":
                    maint_logs = self.skill_pool.maintain_fifo()
                else:
                    maint_logs = self.skill_pool.maintain()

            else:
                evolution_logs = []
                maint_logs = []
                print("⏭ Just testing without evolution.")
            
            swan_data = {
                "iteration": it + 1,
                "epsilon": float(curr_eps),
            }
            
            # 1) 只存 task_baselines（按 env 展开）
            for env_name, baseline in self.task_baselines.items():
                swan_data[f"task_baseline/{env_name}"] = float(baseline)
            
            # 2) 只存本轮 iter_task_returns（按 env 展开）
            for env_name, avg_ret in iter_task_returns.items():
                swan_data[f"avg_return/{env_name}"] = float(avg_ret)
            
            swanlab.log(swan_data, step=it + 1)
            if self.record_tokens:
                # 计算该轮次的平均 ΔPrompt Tokens / Step
                delta_tokens_list = [
                    e.total_added_tokens / max(1, e.step_count)
                    for e in all_iter_experiences
                ]
                avg_delta_tokens = sum(delta_tokens_list) / len(delta_tokens_list) if delta_tokens_list else 0.0
            else:
                avg_delta_tokens = 0.0
            # 记录日志
            log_entry = {
                "iteration": it + 1,
                "avg_return": {k: float(v) for k, v in iter_task_returns.items()},
                "task_baselines": {k: float(v) for k, v in self.task_baselines.items()},
                "global_baseline": float(self.global_baseline),
                "skill_usage_details": iter_skill_usage_logs,
                "delta_prompt_tokens_per_step": float(avg_delta_tokens),  # [新增指标]
                "episodes_data": [
                    {
                        "reward": e.reward,
                        "steps": e.step_count
                    } for e in all_iter_experiences],
                "pool_snapshot": [
                    {
                        "name": o.name,
                        "freq": o.frequency,
                        "maturity": o.maturity,
                        "avg_gain": o.avg_gain,
                        "initiation": o.initiation,  # 新增：记录启动条件
                        "policy": o.policy,  # 新增：记录策略步骤
                        "termination": o.termination  # 新增：记录终止条件
                    }
                    for o in self.skill_pool.get_all()
                ],
                "evolution_details": evolution_logs,
                "maintenance_details": maint_logs,
            }
            training_logs.append(log_entry)
            self._save_training_logs(training_logs, self.log_file_path)
        
        return training_logs
    
    def run_episodes_in_env(self, env_name, epsilon):
        all_returns, all_exps = [], []
        all_usage_histories = []
        for ep in range(self.args.episodes_per_iter):
            reward, used_skills, exp_obj, usage_history = self.run_single_episode(env_name, epsilon)
            all_returns.append(reward)
            all_exps.append(exp_obj)
            all_usage_histories.append(usage_history)
            
            if used_skills:
                m_t = len(used_skills)
                skill_call_counts = Counter(
                    getattr(o, "name", str(o)) for o in used_skills)
                skill_set = {o.name: o for o in used_skills if getattr(o, "name", None)}
                for opt_name, opt in skill_set.items():
                    opt.update_stats(
                        reward=reward,
                        baseline=self.task_baselines[env_name],
                        total_skill_calls_in_traj=m_t,
                        skill_call_count_in_traj=skill_call_counts[opt_name],
                    )

            print(f"✓ Env:{env_name} | Episode {ep + 1} Return: {reward:.4f} ")
        
        return np.mean(all_returns), all_exps, all_usage_histories
    
    def run_single_episode(self, env_name, epsilon):
        is_alfworld = "alfworld" in env_name.lower()
        
        alfworld_history = ""
        
        if is_alfworld:
            env = self.alfworld_env
            obs, info = env.reset()
            state = obs[0]
            admissible_commands = info.get('admissible_commands', [[]])[0]
            
            state_for_prompt = state + '\n>'
        else:
            env = ta.make(env_id=env_name)
            env.reset(num_players=1)
            _, state = env.get_observation()
            admissible_commands = []
            state_for_prompt = state
        print(f"{state_for_prompt}")
        
        done = False
        episode_transitions = []
        active_skill = None
        used_skills_in_ep = []
        skill_usage_history = []  # 新增：用于记录 (步数, Skill名)
        K = int(getattr(self.args, "skill_select_k", 3))
        skill_age = 0  # 当前 active_skill 已连续控制的步数
        MAX_OPTION_AGE = getattr(self.args, "max_skill_age", 8)  # 你想要的K，默认12
        total_reward = 0
        step_count = 0
        max_steps = 50
        total_added_tokens = 0
        current_input_state = state_for_prompt + alfworld_history if is_alfworld else state
        
        while not done and step_count < max_steps:
            
            if self.args.MDP_type == "SMDP" and self.warm_up_flag == False:
                
                if active_skill is not None:
                    if skill_age >= MAX_OPTION_AGE:
                        active_skill = None
                        skill_age = 0
                    else:
                        if self._should_terminate(str(current_input_state), active_skill):
                            active_skill = None
                            skill_age = 0
                
                if active_skill is None:
                    active_skill = self.skill_pool.select_skill(
                        state=str(current_input_state),
                        llm_agent=self.llm_policy,
                        select_type=self.args.select_type,
                        epsilon=epsilon
                    )
                    if active_skill:
                        print(f" [Skill] Activated: {active_skill.name}")
                        used_skills_in_ep.append(active_skill)
                        skill_usage_history.append({
                            "step": step_count,
                            "skill_name": active_skill.name
                        })
                
                skill_text = active_skill.format_for_llm() if active_skill else ""
            else:
                skill_text = ""
                active_skill = None
            
            prompt = self.build_decision_prompt(
                state=current_input_state,
                skill_text=skill_text,
                env_name=env_name,
                admissible_commands=admissible_commands
            )
            if self.record_tokens:
                if hasattr(self.llm_policy, "tokenizer"):
                    base_prompt = self.build_decision_prompt(
                        state=current_input_state,
                        skill_text="",
                        env_name=env_name,
                        admissible_commands=admissible_commands
                    )
                    # 计算两者 Token 数量差值
                    tokens_full = len(self.llm_policy.tokenizer.encode(prompt, add_special_tokens=False))
                    tokens_base = len(self.llm_policy.tokenizer.encode(base_prompt, add_special_tokens=False))
                    total_added_tokens += max(0, tokens_full - tokens_base)
            
            a_t = ''
            for _ in range(5):
                raw_response = self.llm_policy(prompt)
                a_t = self._parse_action_response(raw_response)
                
                if not is_alfworld:
                    if a_t != '' and a_t != '[action]' and a_t != '[d1 d2 d3 d4]' and a_t != '[...]': break
                else:
                    if a_t.strip('[]') in [cmd.strip() for cmd in admissible_commands]:
                        break
            
            transit_dict = {
                "state": str(state),
                "action": str(a_t),
                "skill": active_skill.name if active_skill else "None"
            }
            
            try:
                if is_alfworld:
                    action_str = a_t.strip('[]')
                    obs, scores, dones, infos = env.step([action_str])
                    new_observation = obs[0]
                    total_reward = scores[0]
                    done = dones[0]
                    admissible_commands = infos.get('admissible_commands', [[]])[0]
                    
                    alfworld_history += f" {action_str}\n{new_observation}\n>"
                    
                    print(f"Step: {step_count} | Action: {action_str} | next state: {new_observation} | Done: {done}")
                    
                    state = new_observation  # 更新当前 state 以便 logging
                else:
                    done, step_info = env.step(action=a_t)
                    _, state = env.get_observation()
                
                current_input_state = state_for_prompt + alfworld_history if is_alfworld else state
            except Exception as e:
                print(f"Env Error: {e}")
                done = True
                reason = f"Error: {e}"
            
            transit_dict['done'] = done
            episode_transitions.append(transit_dict)
            skill_age += 1
            step_count += 1
        
        # --- 结算部分保持不变 ---
        if is_alfworld:
            final_reward = float(total_reward)
            feedback = "ALFWorld task completed" if final_reward > 0 else "ALFWorld task failed/timeout"
        else:
            rewards, game_info = env.close()
            final_reward = rewards[0] if rewards else 0.0
            feedback = game_info[0].get('reason', 'No info') if game_info else 'No info'
        
        unique_skill_names = list(set([o.name for o in used_skills_in_ep])) if used_skills_in_ep else []
        

        exp = Experience(
            reward=final_reward,
            trajectory=f"{current_input_state}.\nGame Over!\nFinal reward: {final_reward}.\nFeedback: {feedback}",
            skill=";".join(unique_skill_names) if unique_skill_names else "None",
            env_name=env_name,
            transitions=episode_transitions,
            step_count=step_count,
            total_added_tokens=total_added_tokens
        
        )
        
        swanlab.log({f"episode_return/{env_name}": final_reward})
        return final_reward, used_skills_in_ep, exp, skill_usage_history
    
    def _should_terminate(self, state, skill) -> bool:
        """Return True if the current skill should be stopped.

        Stop an skill when:
          1) Its Termination condition is already satisfied, OR
          2) Its Initiation condition no longer matches the current state.
        """
        if skill is None:
            return True
        
        prompt = f"""[ROLE]
You are a Meta-Controller supervising an AI Agent.

[CURRENT STATE]
{state}

[ACTIVE OPTION]
Name: {skill.name}
Initiation (when to use): {skill.initiation}
Policy (what to do): {skill.policy}
Termination (when to stop): {skill.termination}

[YOUR TASK]
Decide whether the agent should STOP using this skill right now.

Return <status>DONE</status> if EITHER:
- The Termination is already achieved in the CURRENT STATE, OR
- The Initiation is NOT satisfied anymore by the CURRENT STATE.

Otherwise return <status>CONTINUE</status>.

[FORMAT]
Output EXACTLY ONE line, no extra text:
<status>DONE</status>
or
<status>CONTINUE</status>
"""
        
        raw_response = self.llm_policy(prompt)
        status = self._parse_status_response(raw_response)
        
        # Conservative fallback: if parsing fails, keep the skill (do NOT terminate).
        if status not in {"DONE", "CONTINUE"}:
            return False
        return status == "DONE"
    
    def build_decision_prompt(self, state, skill_text, env_name, admissible_commands=None) -> str:
        if 'alfworld' in env_name.lower():
            base_instruction = "You are an embodied agent in a simulated house. Your goal is to complete a specific household task (e.g., put a clean sponge in the cabinet)."
        else:
            base_instruction = ""
        # 处理 ALFWorld 的候选动作
        cmd_hint = ""
        if admissible_commands:
            cmd_hint = "\n[ADMISSIBLE COMMANDS]\n" + "\n".join([f"- {cmd}" for cmd in admissible_commands])
        
        if not skill_text or self.args.MDP_type == "MDP":
            prompt = f"""{base_instruction}
    [CURRENT STATE]
    {state}
    {cmd_hint}
    CRITICAL: Output ONLY the action, max 10 token.\nYou MUST output ONLY one action in format [action]. NO explanations, NO other text. Violation causes error.
[FORMAT]
"""
        else:
            prompt = f"""{base_instruction}
    [CURRENT STATE]
    {state}
    {cmd_hint}

    [ACTIVE OPTION]
    {skill_text}

    [HOW TO USE THE OPTION]
    1) Match: Decide if the skill's Target Situation fits the current state. (Yes/No)
    2) Apply: Execute EACH strategy step one by one. For each step, explicitly reference the relevant part of the CURRENT STATE or feedback history. (no vague wording).
    3) Output: You MUST output ONLY one action in FORMAT. Violation causes error.

    [FORMAT — OUTPUT ONLY]
<think>
match: Yes/No + short reason
apply:
    - Step 1: ...
    - Step 2: ...
</think>
    """
        # 动态添加动作提示
        if 'frozenlake' in env_name.lower():
            prompt += "<action>[direction]</action>\n[ACTION CONSTRAINTS] MUST choose from Valid Actions: [up], [down], [left], [right]."
        elif 'Mastermind-v0' == env_name:
            prompt += "<action>[d1 d2 d3 d4]</action> [ACTION CONSTRAINTS] Numbers 1-6 ONLY. No duplicates (e.g., [1 2 2 3] is invalid). Never repeat a past guess. Available Actions Format: [1 2 3 4] (Exactly 4 digits, space-separated, no commas)."
        elif 'Mastermind-v0-hard' == env_name:
            prompt += "<action>[d1 d2 d3 d4]</action> [ACTION CONSTRAINTS] Numbers 1-8 ONLY, No duplicates (e.g., [1 2 2 3] is invalid). Never repeat a past guess, Available Actions Format: [1 2 3 4] (Exactly 4 digits, space-separated, no commas)."
        elif 'Mastermind-v0-extreme' == env_name:
            prompt += "<action>[d1 d2 d3 d4 d5 d6]</action> [ACTION CONSTRAINTS] Numbers 1-12 ONLY, Never repeat a past guess. Never repeat a past guess, Available Actions Format: [1 2 3 4 5 6] (Exactly 6 digits, space-separated, no commas)."
        elif 'hangman' in env_name.lower():
            prompt += (
                "<action>[letter]</action> or <action>[word]</action>..\n"
                "[ACTION CONSTRAINTS] MUST guess ONE letter, or ONE word.\n"
                "Consider whether guessing a letter would reduce uncertainty more effectively.")
        elif 'alfworld' in env_name.lower():
            prompt += (
                "<action>...</action>\n"
                "The content inside <action> MUST be chose from the ADMISSIBLE COMMANDS."
            )
        prompt += "\nNow produce the final answer following FORMAT exactly:"
        return prompt.strip()
    
    def _parse_action_response(self, llm_output: str) -> str:
        """
        Priority:
        1) <action> ... </action>
        2) [action] ... [/action]
        3) last [...] block
        Return: always "[...]" or "".
        """
        if not llm_output:
            return ""
        
        # 1) <action>...</action>
        m = re.findall(r"<action>\s*(.*?)\s*</action>", llm_output,
                       flags=re.DOTALL | re.IGNORECASE)
        if m:
            content = m[-1].strip()
            return content if content.startswith("[") and content.endswith("]") else f"[{content}]"
        
        # 2) [action]...[/action]
        m = re.findall(r"\[action\]\s*(.*?)\s*\[/action\]", llm_output, flags=re.DOTALL | re.IGNORECASE)
        if m:
            content = m[-1].strip()
            return content if content.startswith("[") and content.endswith("]") else f"[{content}]"
        
        # 3) fallback: last [...]
        m = re.findall(r"\[(.*?)\]", llm_output, flags=re.DOTALL)
        if m:
            return f"[{m[-1].strip()}]"
        
        return ""
    
    def _save_training_logs(self, logs, path):
        """ 实时保存日志 """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Warning] Failed to save logs: {e}")
    
    def _parse_status_response(self, llm_output: str) -> str:
        """
        精准解析终止状态标签 <status>...</status>
        """
        import re
        # 优先匹配 <status> 标签
        tag_match = re.search(r"<status>(.*?)</status>", llm_output, re.DOTALL | re.IGNORECASE)
        if tag_match:
            return tag_match.group(1).strip().upper()
        
        # 兜底逻辑：如果在标签外直接出现了关键词（取最后出现的那个防止解释性文字干扰）
        if "CONTINUE" in llm_output.upper():
            return "CONTINUE"
        return "DONE"