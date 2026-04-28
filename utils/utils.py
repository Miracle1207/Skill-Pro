"""
工具函数模块
包含动作验证、状态提取、状态向量化、余弦相似度计算等辅助函数
"""

import re
import copy
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import os
import random
import numpy as np

import time




def build_log_file_path(args):
    """
    构建训练日志保存路径，自动提取模型名（去掉路径前缀），
    并按时间戳组织文件名。

    输出格式例子：
        outputs/Mastermind-v0_gemma-3-4b-it_llama-3-8b_20251126-153025_training_logs.json
    """
    output_dir = args.output_dir
    env_name = args.env_names.replace(",", "_")
    agent_name = args.agent_name.split('/')[-1]
    ge_model_name = args.ge_model_name.split('/')[-1]
    select_type = args.select_type
    MDP_type = args.MDP_type
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 时间戳
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # 取最后目录名
    agent_name_clean = os.path.basename(os.path.normpath(agent_name))
    ge_model_name_clean = os.path.basename(os.path.normpath(ge_model_name))
    
    # 构建路径
    path_name = f"{env_name}_{MDP_type}_{select_type}_{agent_name_clean}_{ge_model_name_clean}_{timestamp}"
    log_file_path = os.path.join(
        output_dir,
        path_name+"_training_logs.json"
    )
    
    print(f"[Log] 训练日志将保存至: {log_file_path}")
    return path_name, log_file_path


def is_local_model(model_name: str) -> bool:
    """
    Determine whether `model_name` refers to a local LLM.

    Rules:
    1. Absolute paths (starting with "/") → local
    2. Existing filesystem paths → local
    3. Contains typical path keywords (mnt, nas, data, home...) → local
    4. Otherwise → remote model (OpenRouter)
    """
    
    # Absolute path
    if model_name.startswith("/"):
        return True
    
    # Actual path exists
    if os.path.exists(model_name):
        return True
    
    # Common path keywords
    local_keywords = ["mnt", "nas", "data", "home", "cache", "models", "modelscope"]
    if any(k in model_name.lower() for k in local_keywords):
        return True
    
    # Default: remote model
    return False


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for full reproducibility.
    Args:
        seed (int): random seed.
        deterministic (bool): whether to enforce deterministic behavior.
    """

    # ==== Python & Numpy ====
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)



    print(f"[Set Seed] All seeds set to {seed}, deterministic={deterministic}")
    
    




def calculate_avg_reward(total_rewards, total_games):
    """ 计算平均奖励 """
    if total_games == 0:
        return 0.0
    return total_rewards / total_games


def calculate_avg_turns(total_turns, total_games):
    """ 计算平均回合数 """
    if total_games == 0:
        return 0.0
    return total_turns / total_games


def clean_action_format(action: str) -> str:
    """清理动作格式，从包含解释文字的动作中提取纯动作格式"""
    # 首先尝试直接匹配完整的动作格式
    match = re.search(r'\[\s*\d+\s+\d+\s+\d+\s+\d+\s*\]', action)
    if match:
        return match.group(0)
    
    # 如果没有找到完整格式，尝试提取数字并重新构造
    numbers = re.findall(r'\b[1-6]\b', action)
    if len(numbers) >= 4:
        return f"[{numbers[0]} {numbers[1]} {numbers[2]} {numbers[3]}]"
    
    # 如果还是找不到，返回默认动作
    return "[1 2 3 4]"


def validate_action_format(action: str) -> tuple[bool, str]:
    """验证Mastermind动作格式"""
    if not action or not isinstance(action, str):
        return False, "无效的动作格式"
    
    # 检查是否包含自然语言解释
    if any(phrase in action.lower() for phrase in [
        "thank you", "based on", "i will", "let me", "here's", 
        "i'll try", "my guess", "the answer", "i think",
        "considering", "according", "therefore", "so i will"
    ]):
        return False, "动作包含解释文字，请只输出动作格式"
    
    # 使用正则表达式验证Mastermind动作格式
    if not re.search(r'\[\s*\d+\s+\d+\s+\d+\s+\d+\s*\]', action):
        return False, "动作格式不正确，应为 '[数字 数字 数字 数字]' 格式"
    
    # 提取并验证数字
    numbers = re.findall(r'\d+', action)
    if len(numbers) != 4:
        return False, "动作必须包含恰好4个数字"
    
    # 验证数字范围
    for num_str in numbers:
        num = int(num_str)
        if num < 1 or num > 6:
            return False, f"数字 {num} 超出范围 1-6"
    
    return True, ""


def extract_state_key(game_state: Dict[str, Any]) -> str:
    """从游戏状态提取状态键 - 只包含上一轮猜测的数字"""
    if not game_state or "history" not in game_state:
        return "initial"
    
    history = game_state["history"]
    if not history:
        return "initial"
    
    # 只包含上一轮猜测的数字
    last_guess = history[-1].get("guess", [])
    return ",".join(str(x) for x in last_guess)


def create_unified_condition(game_state: Dict[str, Any], task: str) -> str:
    """创建统一的条件描述"""
    if not game_state:
        return f"初始状态；约束：{task}"
    
    # 提取当前状态信息
    history = game_state.get("history", [])
    turn_num = len(history)
    
    if turn_num == 0:
        return f"初始状态，第1回合；约束：{task}"
    
    # 构建历史描述
    history_desc = []
    for i, entry in enumerate(history[-3:], 1):  # 最近3次猜测
        guess = entry.get("guess", [])
        feedback = entry.get("feedback", {})
        black_pegs = feedback.get("black_pegs", 0)
        white_pegs = feedback.get("white_pegs", 0)
        history_desc.append(f"第{turn_num - len(history) + i}次猜测: {guess} -> 黑钉:{black_pegs} 白钉:{white_pegs}")
    
    state_desc = f"第{turn_num}回合，历史: {'; '.join(history_desc)}"
    return f"{state_desc}；约束：{task}"


# ==================== 状态向量化和余弦相似度计算 ====================

class StateVectorizer:
    """状态向量化器 - 将游戏状态转换为数值向量"""
    
    def __init__(self):
        self.feature_dim = 8
    
    def vectorize_state(self, game_state: Any) -> np.ndarray:
        """将游戏状态转换为向量"""
        features = self._extract_state_features(game_state)
        vector = self._features_to_vector(features)
        return self._normalize_vector(vector)
    
    def _extract_state_features(self, game_state: Any) -> Dict[str, Any]:
        """提取状态特征"""
        if not game_state:
            return {}
        
        history = game_state.get('history', [])
        max_turns = game_state.get('max_turns', 20)
        
        features = {
            'turn_count': len(history),
            'code_length': game_state.get('code_length', 4),
            'num_numbers': game_state.get('num_numbers', 6),
            'duplicate_numbers': game_state.get('duplicate_numbers', False),
            'last_black_pegs': 0,
            'last_white_pegs': 0,
            'total_black_pegs': 0,
            'total_white_pegs': 0,
            'progress_ratio': len(history) / max_turns
        }
        
        if history:
            last_guess = history[-1]
            features['last_black_pegs'] = last_guess.get('black', 0)
            features['last_white_pegs'] = last_guess.get('white', 0)
            
            for guess in history:
                features['total_black_pegs'] += guess.get('black', 0)
                features['total_white_pegs'] += guess.get('white', 0)
        
        return features
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """将特征转换为向量"""
        vector = np.array([
            features.get('turn_count', 0),
            features.get('code_length', 4),
            features.get('num_numbers', 6),
            float(features.get('duplicate_numbers', False)),
            features.get('last_black_pegs', 0),
            features.get('last_white_pegs', 0),
            features.get('total_black_pegs', 0),
            features.get('total_white_pegs', 0),
            features.get('progress_ratio', 0.0)
        ])
        return vector
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """向量归一化"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector


class CosineSimilarityCalculator:
    """余弦相似度计算器"""
    
    def __init__(self, similarity_threshold: float = 1):
        self.similarity_threshold = similarity_threshold
        self.state_vectorizer = StateVectorizer()
    
    def _vectorize_recent_guess(self, game_state: Any) -> Optional[np.ndarray]:
        """将最近一次猜测向量化为位置敏感 one-hot 向量。
        长度 = code_length * num_numbers；第 pos 位的数字 d(1..num_numbers) 映射到
        index = pos*num_numbers + (d-1)。无历史或不合法时返回 None。
        """
        try:
            if not isinstance(game_state, dict):
                return None
            history = game_state.get('history', []) or []
            if not history:
                return None
            last_guess = history[-1].get('guess', []) or []
            code_length = int(game_state.get('code_length', 4))
            num_numbers = int(game_state.get('num_numbers', 6))
            if len(last_guess) != code_length:
                return None
            vec_len = code_length * num_numbers
            vec = np.zeros(vec_len, dtype=float)
            for pos, val in enumerate(last_guess):
                if not isinstance(val, int):
                    return None
                if val < 1 or val > num_numbers:
                    return None
                idx = pos * num_numbers + (val - 1)
                vec[idx] = 1.0
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            return None
    
    def calculate_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return dot_product / norm_product if norm_product > 0 else 0.0
        except Exception as e:
            print(f"计算余弦相似度失败: {e}")
            return 0.0
    
    def find_most_similar_guidance(self, current_state: Any, guidance_pool) -> Tuple[Optional[Any], float]:
        """找到与当前状态最相似的指导：比较最近一次猜测的 one-hot 位置向量。"""
        try:
            current_guess_vec = self._vectorize_recent_guess(current_state)
            if current_guess_vec is None:
                return None, 0.0
            best_guidance = None
            best_similarity = 0.0
            
            for guidance in guidance_pool.get_all_guidances():
                # 需要 original_state 以获取该指导生成时的最近猜测
                if not hasattr(guidance, 'original_state') or guidance.original_state is None:
                    continue
                guidance_guess_vec = self._vectorize_recent_guess(guidance.original_state)
                if guidance_guess_vec is None:
                    continue
                similarity = self.calculate_cosine_similarity(current_guess_vec, guidance_guess_vec)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_guidance = guidance
            
            if best_similarity >= self.similarity_threshold:
                return best_guidance, best_similarity
            return None, 0.0
        except Exception as e:
            print(f"查找最相似指导失败: {e}")
            return None, 0.0


# 创建全局实例
state_vectorizer = StateVectorizer()
cosine_similarity_calculator = CosineSimilarityCalculator(similarity_threshold=1)
