"""
Pool Management Module
Contains GuidancePool and ExperiencePool classes.
"""

from typing import Dict, List, Optional, Any
from data_structures import Experience
import random
import heapq  # For maintaining Top-K in Golden Pool


class ExperiencePool:
    """Standard FIFO Experience replay buffer."""
    
    def __init__(self, max_size: int = 1000):
        self.experiences: List[Experience] = []
        self.max_size = max_size
    
    def add(self, experience: Experience):
        """Add a new experience to the pool."""
        if len(self.experiences) >= self.max_size:
            # Remove the oldest experience
            self.experiences.pop(0)
        self.experiences.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Randomly sample a batch of experiences.
        """
        if len(self.experiences) == 0:
            return []
        
        if batch_size >= len(self.experiences):
            # If not enough experiences, return all available ones
            return random.sample(self.experiences, len(self.experiences))
        
        return random.sample(self.experiences, batch_size)


class GoldenExperiencePool:
    """
    [NEW] Golden Experience Pool
    Retains only the Top-K high-reward experiences.
    Acts as a proxy for 'High Advantage' behaviors for verification.
    """
    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.pool: List[Any] = []  # Stores tuples: (reward, unique_id, experience)
    
    def add(self, experience: Experience):
        """
        Maintains Top-K mechanism:
        - If pool is not full, add directly.
        - If full, replace the worst experience if the new one is better.
        """
        # If experience lacks structured transition data, it cannot be used for verification
        if not experience.transitions:
            return

        # Use id(experience) as a tie-breaker to avoid comparison errors with Experience objects
        item = (experience.reward, id(experience), experience)

        if len(self.pool) < self.max_size:
            heapq.heappush(self.pool, item)
        else:
            # Check against the smallest reward (heap root)
            min_reward = self.pool[0][0]
            if experience.reward > min_reward:
                heapq.heappop(self.pool)
                heapq.heappush(self.pool, item)
    
    def get_all(self, limit: int = None) -> List[Experience]:
        """Returns all Golden Experiences (unordered), optionally limited to top-k."""
        # Extract all items
        all_items = self.pool
        
        # If a limit is specified and is smaller than current pool size,
        # we sort to return the best ones.
        if limit is not None and limit < len(all_items):
            # Sort by reward descending (index 0 of the tuple)
            sorted_items = sorted(all_items, key=lambda x: x[0], reverse=True)
            return [item[2] for item in sorted_items[:limit]]
            
        return [item[2] for item in all_items]
    
    def get_transitions(self) -> List[Dict[str, Any]]:
        """
        Flattens all experiences into a list of (state, action) transitions.
        Used for batch score calculation.
        """
        all_trans = []
        for _, _, exp in self.pool:
            all_trans.extend(exp.transitions)
        return all_trans