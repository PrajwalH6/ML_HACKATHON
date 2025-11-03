"""Pure HMM baseline agent (no learning)."""
from typing import Dict, Set

class HMMBaselineAgent:
    """Agent that uses only HMM oracle probabilities."""
    
    def __init__(self):
        pass
    
    def select_action(self, hmm_probs: Dict[str, float], valid_actions: Set[str]) -> str:
        """Select letter with highest HMM probability."""
        valid_probs = {letter: hmm_probs.get(letter, 0.0) 
                      for letter in valid_actions}
        
        if not valid_probs or sum(valid_probs.values()) == 0:
            # Fallback to most common letters
            common_order = 'etaoinshrdlcumwfgypbvkjxqz'
            for letter in common_order:
                if letter in valid_actions:
                    return letter
            return list(valid_actions)[0]
        
        return max(valid_probs, key=valid_probs.get)
