"""HMM emission probability computation."""
import numpy as np
from collections import defaultdict
import pickle
from pathlib import Path
from typing import Dict, List

class EmissionBuilder:
    """Build per-position letter emission probabilities."""
    
    def __init__(self, smoothing_alpha=1.0):
        self.alpha = smoothing_alpha
        self.emissions = {}  # {length: [{letter: prob}, ...]}
        
    def build_from_corpus(self, words_by_length: Dict[int, List[str]]) -> Dict:
        """Build per-position emission probabilities for each word length."""
        print("Building HMM emissions...")
        
        for length, words in words_by_length.items():
            if length == 0:
                continue
            self.emissions[length] = self._compute_emissions(words, length)
            print(f"  Length {length}: {len(words)} words processed")
        
        return self.emissions
    
    def _compute_emissions(self, words: List[str], length: int) -> List[Dict[str, float]]:
        """Compute smoothed per-position letter probabilities."""
        # Count letters at each position
        position_counts = [defaultdict(int) for _ in range(length)]
        
        for word in words:
            for pos, char in enumerate(word):
                position_counts[pos][char] += 1
        
        # Apply Laplace smoothing and normalize
        emissions = []
        for pos in range(length):
            counts = position_counts[pos]
            total = sum(counts.values()) + self.alpha * 26
            
            probs = {}
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                probs[letter] = (counts[letter] + self.alpha) / total
            
            emissions.append(probs)
        
        return emissions
    
    def save(self, filepath='models/hmm/emissions.pkl'):
        """Save emission parameters."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.emissions, f)
        
        print(f"\nEmissions saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/hmm/emissions.pkl') -> Dict:
        """Load emission parameters."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
