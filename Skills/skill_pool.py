# Skills/skill_pool.py
import numpy as np
import re
from typing import List, Any, Optional, Dict
from data_structures import Skill
from utils.encode import encode_text
import random
import numpy as np
import re
import math
from collections import defaultdict

class SkillPool:
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.skills: List[Skill] = []
        self._skill_embs: Dict[str, np.ndarray] = {}
        self._initialize_seeds()
    
    def get_all(self):
        """
        返回池中所有的 Skill 对象
        """
        return self.skills
    

    def _initialize_seeds(self):
        self.skills.append(Skill(
            name="StructuredCoT",
            initiation="When a decision must be made based on multiple constraints or past feedback.",
            policy=[
                "Restate the immediate goal of the task in one sentence.",
                "List all hard constraints implied by the current state and feedback history.",
                "Summarize the key information from previous actions and feedback that affects the decision.",
                "Compare the main candidate actions step by step under these constraints.",
                "Select the single action that best satisfies the constraints and goal."
            ],
            termination="A single concrete action is selected and output."
        ))
        self.skills.append(Skill(
            name="ReActDecision",
            initiation="When the environment provides feedback after each action and past feedback should influence the next decision.",
            policy=[
                "Interpret the most recent feedback and explain what it implies about the environment.",
                "Update your belief about which actions or outcomes are more or less likely.",
                "Choose the next action that best exploits or tests this updated belief.",
                "Output only the chosen action."
            ],
            termination="The next action is selected based on the updated belief."
        ))
        self.skills.append(Skill(
            name="HypothesisElimination",
            initiation="When multiple past guesses with feedback exist and more than one hidden hypothesis remains plausible.",
            policy=[
                "Enumerate the main plausible hypotheses consistent with all past feedback.",
                "Eliminate hypotheses that contradict any feedback in the history.",
                "Identify what information is still uncertain among the remaining hypotheses.",
                "Choose an action that best distinguishes among these remaining hypotheses."
            ],
            termination="An action aimed at reducing hypothesis uncertainty is selected."
        ))
        self.skills.append(Skill(
            name="SelfConsistencyCheck",
            initiation="When the action must strictly satisfy known rules or historical constraints.",
            policy=[
                "Propose a candidate action.",
                "Check whether this action violates any known rules or past feedback.",
                "If a violation is found, revise the action to remove the violation.",
                "Repeat the check until no violation remains.",
                "Output the final consistent action."
            ],
            termination="An action that passes all self-consistency checks is produced."
        ))
        self.skills.append(Skill(
            name="ExploreExploitArbitration",
            initiation="When several reasonable actions exist and the agent must balance exploration and exploitation.",
            policy=[
                "Assess whether recent actions have significantly reduced uncertainty or improved confidence.",
                "If recent information gain has been low, favor an exploratory action that differs from recent ones.",
                "If confidence is high, favor the action most consistent with past successful feedback.",
                "Explicitly decide whether the current step is exploration or exploitation.",
                "Output the chosen action."
            ],
            termination="One action is selected with a clear exploration or exploitation rationale."
        ))
        self.skills.append(Skill(
            name="StrategicPlanning",
            initiation="At the very beginning of the game (Turn 1) when no previous feedback exists.",
            policy=[
                "Choose a mathematically diverse starting guess (e.g., all unique or a 2+2 pattern).",
                "Establish the initial search boundaries based on the allowed digit range.",
                "Set the logic for tracking historical moves."
            ],
            termination="The first valid move is submitted and the environment returns initial feedback."
        ))


        for opt in self.skills:
            self._update_emb_cache(opt)
    
    def _update_emb_cache(self, opt: Skill):
        """预计算并将 Skill 的启动条件向量化"""
        if opt.name not in self._skill_embs:
            self._skill_embs[opt.name] = encode_text(opt.initiation)
    
    def add_skill(self, opt: Skill):
        """添加新 Skill 并同步更新缓存"""
        if opt not in self.skills:
            self.skills.append(opt)
            self._update_emb_cache(opt)

    def select_skill(
            self,
            state: str,
            llm_agent: Any = None,
            select_type: str = "similarity",
            epsilon: float = 0.1,
            topk: int = 3,  # NEW: top-k for llm
            min_gain: float = 0.0  # NEW: require avg_gain > min_gain (default 0)
    ):
        """
        带 Epsilon-Greedy 探索的双模选择器
        epsilon: 探索概率，建议在训练早期设为 0.2-0.3，后期减小。
        """
        if not self.skills:
            return None
    
        # --- Epsilon-Greedy 探索分支 ---
        if random.random() < epsilon:
            choices = self.skills + [None]
            explored_opt = random.choice(choices)
            return explored_opt
    
        if select_type == "llm_model" and llm_agent is not None:
            return self._select_by_llm(state, llm_agent)
    
        # NEW: LLM topK + avg_gain rerank
        if select_type == "llm_topk_gain" and llm_agent is not None:
            return self._select_by_llm_topk_gain(
                state,
                llm_agent,
                k=topk,
                min_gain=min_gain
            )
        
        if select_type == "llm_topk_lcb" and llm_agent is not None:
            return self.select_skill_llm_topk_lcb_simple(
                state, llm_agent, k=topk, beta=0.2, min_lcb=0.0
            )

        return self._select_by_similarity(state, threshold=0.5)

    def select_skill_llm_topk_lcb_simple(
            self,
            state: str,
            llm_agent: Any,
            k: int = 3,
            beta: float = 0.2,
            min_lcb: float = 0.0,
            warmup_maturity: int = 3,
    ):
        """
        LLM nominates top-k most relevant skills;
        system picks the one with highest LCB(avg_gain, freq).
        """
        k = min(max(1, int(k)), len(self.skills))
    
        skills_desc = "\n".join(
            f"- {o.name}: {getattr(o, 'description', o.initiation)}"
            for o in self.skills
        )
    
        prompt = f"""You are an expert skill selector.

    [ENVIRONMENT STATE]
    {state}

    [AVAILABLE SKILLS]
    {skills_desc}
    - NONE

    [INSTRUCTIONS]
    Select up to {k} skills that are MOST relevant and helpful.
    Output ONLY XML lines:
    <choice>SkillName</choice>
    """
        resp = llm_agent(prompt)
    
        names = [n.strip() for n in re.findall(r"<choice>\s*(.*?)\s*</choice>", resp, re.I)]
        if not names or any(n.upper() == "NONE" for n in names):
            return None
    
        name_set = {n.lower() for n in names}
        nominees = [o for o in self.skills if o.name.lower() in name_set]
        if not nominees:
            return None
    
        # conservative gain via LCB
        t = max(int(getattr(o, "maturity", 0)) for o in self.skills)
        t = max(t, 1)
    
        def lcb(o):
            mu = float(getattr(o, "avg_gain", 0.0))
            n = max(int(getattr(o, "frequency", 1)), 1)
            return mu - float(beta) * math.sqrt(math.log1p(t) / n)
    
        best = max(nominees, key=lcb)
    
        if int(getattr(best, "maturity", 0)) < int(warmup_maturity):
            return best
    
        return best if lcb(best) > float(min_lcb) else None

    def _select_by_similarity(self, state: str, threshold: float = 0.35) -> Optional[Skill]:
        """
        改进后的语义相似度选择器
        """
        # 1. 将当前状态向量化 (仅计算一次)
        state_vec = encode_text(state)
        state_norm = np.linalg.norm(state_vec) + 1e-9
        
        best_opt = None
        max_sim = -1.0
        
        # 2. 遍历池中已缓存的向量
        for opt in self.skills:
            # 直接从内部字典取向量，无需重新计算
            init_vec = self._skill_embs.get(opt.name)
            if init_vec is None:
                # 兜底：如果缓存缺失则补全
                init_vec = encode_text(opt.initiation)
                self._skill_embs[opt.name] = init_vec
            
            # 计算余弦相似度
            dot_val = np.dot(state_vec, init_vec)
            init_norm = np.linalg.norm(init_vec) + 1e-9
            sim = dot_val / (state_norm * init_norm)
            
            if sim > max_sim:
                max_sim = sim
                best_opt = opt
        
        # 3. 阈值判定：如果最高相似度都太低，返回 None（即不使用任何 Skill）
        if max_sim < threshold:
            # print(f" [Similarity] No matching skill (Max Sim: {max_sim:.4f} < {threshold})")
            return None
        
        return best_opt

    def _select_by_llm(self, state: str, llm_agent: Any, min_gain=-0.015) -> Skill:
        """
        针对 9B 模型优化的决策 Prompt：
        1. 引入 Thought 过程（思维链）
        2. 强化“最优决策”目标
        3. 严格限定输出格式
        """
        skill_names = [o.name for o in self.skills]
        # 假设 Skill 对象有 description 属性，如果没有，请沿用 initiation
        skills_desc = "\n".join([f"- {o.name}: {getattr(o, 'description', o.initiation)}" for o in self.skills])
    
        prompt = f"""You are an expert strategic reasoning agent. Your goal is to analyze the environment state and select the single most suitable Skill to achieve an optimal outcome.

    [ENVIRONMENT STATE]
    {state}

    [AVAILABLE skills]
    {skills_desc}
    - NONE: Choose this if no specialized strategy is required or if the state is simple.

    [INSTRUCTIONS]
    1. Analyze the current [ENVIRONMENT STATE]. Identify the core challenges or needs.
    2. Evaluate each [AVAILABLE SKILL] against the state. Determine which one provides the highest strategic value.
    3. If no skill provides a clear advantage, select "NONE".

    [OUTPUT FORMAT]
    Output ONLY in XML tags, for example: <choice>SkillName</choice> or <choice>NONE</choice>.

    [DECISION]
    """
        response = llm_agent(prompt).strip()
    
        # 1. 检查 NONE
        if "NONE" in response.upper() and "<choice>NONE</choice>" in response.upper():
            return None

        best = None

        match = re.search(r"<choice>\s*(.*?)\s*</choice>", response, re.IGNORECASE | re.DOTALL)
        if match:
            choice = match.group(1).strip()
            for opt in self.skills:
                if opt.name.lower() == choice.lower():
                    best = opt
        else:
            for opt in self.skills:
                if f"<{opt.name}>" in response or opt.name.lower() in response.lower():
                    best = opt
        if getattr(best, "avg_gain", -0.1) > min_gain or getattr(best, "maturity", 0.0) < 3:
            return best
        else:
            return None


    def _select_by_llm_topk_gain(
            self,
            state: str,
            llm_agent: Any,
            k: int = 5,
            min_gain = 0.0
    ):
        """
        LLM 提名 K 个有助于当前 decision making 的 skill，
        系统在其中选择 avg_gain 最大且 > 0 的。
        """
        if not self.skills:
            return None
    
        k = min(k, len(self.skills))
    
        skills_desc = "\n".join(
            f"- {o.name}: {getattr(o, 'description', o.initiation)}"
            for o in self.skills
        )
    
        prompt = f"""You are an expert selector. Select the TOP-{k} most suitable skills for the state.

[ENVIRONMENT STATE]
{state}

[AVAILABLE SKILLS]
{skills_desc}
- NONE: Choose this if no skill is suitable.

[INSTRUCTIONS]
- Return exactly {k} items if possible (unless you choose NONE).
- Each item MUST be an skill name from the list above, or NONE.
- Do NOT invent new names.

    [OUTPUT FORMAT]
    Output ONLY in XML tags:
    <choice>SkillName</choice>
    <choice>SkillName</choice>
    """
    
        resp = llm_agent(prompt)
    
        # --- parse <choice> ---
        names = re.findall(r"<choice>\s*(.*?)\s*</choice>", resp, re.I)
    
        # NONE
        if any(n.strip().upper() == "NONE" for n in names):
            return None
    
        # map to skills
        cand = [
            o for o in self.skills
            if o.name.lower() in {n.strip().lower() for n in names}
        ]
    
        if not cand:
            return None
    
        # rerank by avg_gain
        best = max(cand, key=lambda o: getattr(o, "avg_gain", 0.0))
        return best if getattr(best, "avg_gain", 0.0) > min_gain else None

    def update_pool_status(self):
        """在每一轮 Iteration 结束时，必须统一调用此函数"""
        for opt in self.skills:
            opt.increment_maturity()

    def _semantic_dedup_inplace(
            self,
            skills,
            score_fn,
            log_remove,
            stage: str = "semantic-duplicate",
            thr: float = 0.95,
    ):
        """In-place semantic dedup by clustering (pairwise cosine >= thr). Keep best score per cluster."""
    
        def cos(a, b):
            dot = na = nb = 0.0
            for x, y in zip(a, b):
                x = float(x);
                y = float(y)
                dot += x * y
                na += x * x
                nb += y * y
            return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-12)
    
        opts = list(skills)
        n = len(opts)
        if n <= 1:
            return
    
        # embeddings
        emb = [None] * n
        for i, o in enumerate(opts):
            try:
                self._update_emb_cache(o)
                emb[i] = self._skill_embs.get(o.name)
            except Exception:
                emb[i] = None
    
        # adjacency
        adj = [set() for _ in range(n)]
        for i in range(n):
            ei = emb[i]
            if ei is None:
                continue
            for j in range(i + 1, n):
                ej = emb[j]
                if ej is None:
                    continue
                if cos(ei, ej) >= thr:
                    adj[i].add(j);
                    adj[j].add(i)
    
        # connected components -> remove all but best
        seen = [False] * n
        rm = []
        for i in range(n):
            if seen[i]:
                continue
            st, comp = [i], []
            seen[i] = True
            while st:
                u = st.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        st.append(v)
        
            if len(comp) <= 1:
                continue

            protect_maturity = 3  # 新生 skill 的保护期

            best = max(comp, key=lambda k: score_fn(opts[k]))
            for k in comp:
                if k == best:
                    continue
                if int(getattr(opts[k], "maturity", 0)) < protect_maturity:
                    continue  # protect newborn skill
                rm.append(opts[k])

        # apply removals
        for o in rm:
            if o in skills:
                skills.remove(o)
                log_remove(o, stage)
                if hasattr(self, "_skill_embs"):
                    self._skill_embs.pop(o.name, None)

    def maintain(
            self,
            maturity_max: int = 2,
            n_min: int = 10,
            mid_freq: int = 80,
            neg_gain_thr: float = -0.001,
            new_cap: int = 2,
            maturity_base_dedup: int = 4,
    ):
        """
        Maintain skill pool with minimal, human-readable removal logs.

        Returns:
            maint_logs: List[dict], each with fields:
                - stage: removal category (human-readable)
                - name, freq, maturity
        """
    
        # ---------- human-readable stages ----------
        STAGE_INVALID = "invalid skill"
        STAGE_NEGATIVE = "Negative LCB"
        STAGE_DUPLICATE = "duplicate skill"
        STAGE_OUTDATED = "outdated variant"
        STAGE_OVERFLOW = "capacity overflow"
    
        maint_logs = []
    
        def log_remove(o, stage: str):
            maint_logs.append({
                "stage": stage,
                "name": o.name,
                "avg_gain": getattr(o, "avg_gain", 0),
                "freq": int(getattr(o, "frequency", 0)),
                "maturity": int(getattr(o, "maturity", 0)),
            })

        t = max(int(getattr(o, "maturity", 0)) for o in self.skills) if self.skills else 1

        def lcb(o):
            mu = float(getattr(o, "avg_gain", 0.0))
            n = max(int(getattr(o, "frequency", 1)), 1)
            return mu - 0.2 * math.sqrt(math.log1p(t) / n)

        def base_name(name: str) -> str:
            return re.sub(r"_v\d+$", "", name or "")
    
        # ============================================================
        # (1) Drop invalid / empty skills
        # ============================================================
        kept = []
        for o in self.skills:
            if o.name in (None, "None"):
                log_remove(o, STAGE_INVALID)
            elif not (o.initiation or o.termination or o.policy):
                log_remove(o, STAGE_INVALID)
            else:
                kept.append(o)
        self.skills = kept

        # ============================================================
        # (3) Deduplicate same-name skills
        # ============================================================
        by_name = defaultdict(list)
        for o in self.skills:
            by_name[o.name].append(o)
    
        for lst in by_name.values():
            if len(lst) > 1:
                lst.sort(key=lcb, reverse=True)
                for o in lst[1:]:
                    if len(self.skills) <= self.max_size:
                        break
                    self.skills.remove(o)
                    log_remove(o, STAGE_DUPLICATE)
    

        self._semantic_dedup_inplace(
            self.skills,
            score_fn=lcb,
            log_remove=log_remove,
            stage="semantic-duplicate",
            thr=0.95,
        )

        # ============================================================
        # (4) Deduplicate outdated versions (mature only)
        # ============================================================
        by_base = defaultdict(list)
        for o in self.skills:
            if o.maturity >= maturity_base_dedup:
                by_base[base_name(o.name)].append(o)
    
        for lst in by_base.values():
            if len(lst) > 1:
                lst.sort(key=lcb, reverse=True)
                for o in lst[1:]:
                    if len(self.skills) <= self.max_size:
                        break
                    self.skills.remove(o)
                    log_remove(o, STAGE_OUTDATED)

        # ============================================================
        # (2) Prune consistently harmful (negative LCB)
        # ============================================================
        

        while len(self.skills) > self.max_size:
            candidates = [
                o for o in self.skills
                if o.maturity > maturity_max
                   and o.frequency >= n_min
                   and lcb(o) < neg_gain_thr
            ]
            if not candidates:
                break
            victim = min(candidates, key=lcb)
            self.skills.remove(victim)
            log_remove(victim, STAGE_NEGATIVE)

        if len(self.skills) <= self.max_size:
            print(f"[Pool] Size OK ({len(self.skills)}/{self.max_size}), skip capacity trim.")
            return maint_logs
        
        # ============================================================
        # (5) Capacity-based structure trim (old > mid > new)
        # ============================================================
        veterans = [o for o in self.skills if o.frequency >= mid_freq]
        mid_age = [o for o in self.skills if n_min <= o.frequency < mid_freq]
        newcomers = [o for o in self.skills if o.frequency < n_min]
    
        veterans.sort(key=lcb, reverse=True)
        mid_age.sort(key=lcb, reverse=True)
        newcomers.sort(key=lcb, reverse=True)
    
        final = veterans + mid_age + newcomers
        final = final[: self.max_size]
    
        final_ids = {id(o) for o in final}
        for o in self.skills:
            if id(o) not in final_ids:
                log_remove(o, STAGE_OVERFLOW)
    
        self.skills = final
        print(f"[Pool] Maintenance complete. Remaining: {len(self.skills)}")
        return maint_logs

    @classmethod
    def from_json(cls, file_path: str, iteration: int = -1, max_size: int = 10):
        """
        从本地 JSON 日志文件中加载指定轮次的 Skill Pool 快照
        """
        import json
        from data_structures import Skill
    
        instance = cls(max_size=max_size)
        instance.skills = []  # 清空种子 Skill
    
        with open(file_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
            target_entry = None
            if iteration == -1:
                target_entry = logs[-1]  # 默认最后一次
            else:
                target_entry = logs[iteration-1]
        
            if target_entry is None:
                raise ValueError(f"未找到 Iteration {iteration}")
        
            snapshot = target_entry['pool_snapshot']
            for opt_data in snapshot:
                opt = Skill(
                    name=opt_data['name'],
                    initiation=opt_data.get('initiation', ""),
                    policy=opt_data.get('policy', []),
                    termination=opt_data.get('termination', ""),
                    frequency=opt_data.get('freq', 0),
                    avg_gain=opt_data.get('avg_gain', 0.0),
                    maturity=opt_data.get('maturity', 0)
                )
                instance.add_skill(opt)
    
        print(f"成功从 {file_path} 加载 Iteration {target_entry.get('iteration')} 的 Pool")
        return instance

    def maintain_fifo(self):
        """
        Ablation version: Maintains the skill pool using a simple FIFO strategy.

        Logic:
        1. No performance-based pruning (no LCB check).
        2. No semantic or version deduplication.
        3. If size exceeds max_size, remove the oldest skills first.

        Returns:
            maint_logs: List[dict] tracking removed skills for comparison.
        """
        STAGE_OVERFLOW = "FIFO overflow (oldest removed)"
        maint_logs = []
    
        target_size = self.max_size
    
        def log_remove(o, stage: str):
            maint_logs.append({
                "stage": stage,
                "name": o.name,
                "avg_gain": getattr(o, "avg_gain", 0),
                "freq": int(getattr(o, "frequency", 0)),
                "maturity": int(getattr(o, "maturity", 0)),
            })
    
        # 检查是否溢出
        if len(self.skills) <= target_size:
            return maint_logs
    
        # 计算需要移除的数量
        num_to_remove = len(self.skills) - target_size
    
        # 假设 self.skills 是按时间顺序排列的 (新加入的在 list 末尾)
        # FIFO 移除列表开头的元素
        to_be_removed = self.skills[:num_to_remove]
        self.skills = self.skills[num_to_remove:]
    
        for o in to_be_removed:
            log_remove(o, STAGE_OVERFLOW)
    
        print(f"[Ablation-Pool] FIFO complete. Removed {num_to_remove} oldest skills. Remaining: {len(self.skills)}")
        return maint_logs