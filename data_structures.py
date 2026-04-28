"""
Data Structure Definitions
Contains core data structures like Guidance and Experience.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import numpy as np


@dataclass
class Skill:
    name: str = "NewSkill"
    initiation: str = ""
    policy: List[str] = field(default_factory=list)
    termination: str = ""
    
    # --- 演化统计指标 ---
    frequency: int = 0  # 被调用的频次 (仅选中时更新)
    avg_gain: float = 0.0  # 相对基准的平均收益提升率 (仅选中时更新)
    total_gain: float = 0.0
    maturity: int = 0  # 成熟度：生存的迭代轮次 (每轮迭代+1)
    success_count: int = 0
    last_evolved_iter: int = 0
    
    parent_id: str = ""
    version: int = 0


    def update_stats(
            self,
            reward: float,
            baseline: float,
            total_skill_calls_in_traj: int,
            skill_call_count_in_traj: int = 1,
    ):
        """
        Update skill statistics using per-trajectory uniform credit assignment.

        Args:
            reward: R(τ_t)
            baseline: \bar{R}_t
            total_skill_calls_in_traj: m_t = |C(τ_t)|
            skill_call_count_in_traj: c_t(ω), default = 1
        """
    
        # --- trajectory-level advantage ---
        advantage = reward - baseline
    
        # --- per-call credit g_t ---
        if total_skill_calls_in_traj > 0:
            per_call_gain = advantage / total_skill_calls_in_traj
        else:
            per_call_gain = 0.0
    
        # --- accumulate G_t and N_t ---
        self.total_gain += skill_call_count_in_traj * per_call_gain
        self.frequency += skill_call_count_in_traj
    
        # --- online avg_gain ---
        denom = max(1, self.frequency)
        self.avg_gain = self.total_gain / denom
    
        # --- optional success counter (kept consistent with paper semantics) ---
        if advantage > 0:
            self.success_count += 1

    def increment_maturity(self):
        """每一轮迭代结束时调用，无论是否被选中"""
        self.maturity += 1
    
    def calculate_maintenance_score(self, total_freq: int) -> float:
        """
        计算用于清理的评分：score = (freq / sum(freq)) * avg_gain
        """
        freq_weight = (self.frequency / total_freq) if total_freq > 0 else 0
        return freq_weight * self.avg_gain
    
    def calculate_score(self, alpha: float, beta: float) -> float:
        """ Score = alpha * frequency + beta * avg_gain """
        return alpha * self.frequency + beta * self.avg_gain
    
    def format_for_llm(self) -> str:
        """
        极简描述格式，减少 LLM 的 Token 消耗并提高指令遵循度
        """
        policy_str = "\n".join([f"- {step}" for step in self.policy])
        return (
            f"Skill Name: {self.name}\n"
            f"Initiation (When to use): {self.initiation}\n"
            f"Strategy Steps:\n{policy_str}\n"
            f"Termination (When to stop): {self.termination}"
        )
    
    
    
@dataclass
class Experience:
    # Skills: str
    reward: float
    skill: str
    trajectory: str
    env_name: str = ""  # [新增] 记录来自哪个任务
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    total_added_tokens: int = 0

