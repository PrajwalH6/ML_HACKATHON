"""HMM-based letter prediction oracle."""
import numpy as np
from collections import Counter
from typing import Dict, List, Set

class HMMOracle:
    """Predict next letter probabilities using HMM emissions."""
    
    def __init__(self, emissions: Dict, words_by_length: Dict):
        self.emissions = emissions
        self.words_by_length = words_by_length
        
    def get_letter_probs(self, mask: str, guessed_letters: Set[str]) -> Dict[str, float]:
        """
        Compute letter probabilities given current game state.
        
        Args:
            mask: Current word mask (e.g., "_pp_e")
            guessed_letters: Set of already guessed letters
        
        Returns:
            dict: {letter: probability} for unguessed letters
        """
        word_length = len(mask)
        
        # Filter candidates matching mask
        candidates = self._filter_candidates(mask, guessed_letters, word_length)
        
        if not candidates:
            # Fallback to position-based probs if no candidates
            return self._position_based_probs(mask, guessed_letters, word_length)
        
        # Aggregate letter frequencies from candidates
        letter_counts = Counter()
        for word in candidates:
            for i, char in enumerate(word):
                if mask[i] == '_':  # Only count letters in blank positions
                    letter_counts[char] += 1
        
        # Normalize to probabilities
        total = sum(letter_counts.values())
        if total == 0:
            return self._uniform_probs(guessed_letters)
        
        probs = {char: count / total 
                 for char, count in letter_counts.items()
                 if char not in guessed_letters}
        
        # Fill in zeros for letters not in candidates
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter not in guessed_letters and letter not in probs:
                probs[letter] = 0.0
        
        return probs
    
    def _filter_candidates(self, mask: str, guessed_letters: Set[str], 
                          word_length: int) -> List[str]:
        """Filter words matching current mask and constraints."""
        candidates = []
        word_list = self.words_by_length.get(word_length, [])
        
        for word in word_list:
            if self._matches_mask(word, mask, guessed_letters):
                candidates.append(word)
        
        return candidates
    
    @staticmethod
    def _matches_mask(word: str, mask: str, guessed_letters: Set[str]) -> bool:
        """Check if word matches mask and guessed constraints."""
        if len(word) != len(mask):
            return False
        
        for w_char, m_char in zip(word, mask):
            # Check revealed positions match
            if m_char != '_' and w_char != m_char:
                return False
            # Check blank positions don't have guessed letters
            if m_char == '_' and w_char in guessed_letters:
                return False
        
        # Check word doesn't contain wrong guesses
        for char in guessed_letters:
            if char not in mask and char in word:
                return False
        
        return True
    
    def _position_based_probs(self, mask: str, guessed_letters: Set[str], 
                             word_length: int) -> Dict[str, float]:
        """Fallback: use HMM emissions for blank positions."""
        if word_length not in self.emissions:
            return self._uniform_probs(guessed_letters)
        
        position_emissions = self.emissions[word_length]
        letter_scores = {chr(ord('a') + i): 0.0 for i in range(26)}
        
        blank_positions = [i for i, char in enumerate(mask) if char == '_']
        
        for pos in blank_positions:
            if pos < len(position_emissions):
                for letter, prob in position_emissions[pos].items():
                    if letter not in guessed_letters:
                        letter_scores[letter] += prob
        
        # Normalize
        total = sum(letter_scores.values())
        if total > 0:
            letter_scores = {k: v / total for k, v in letter_scores.items()}
        
        return letter_scores
    
    @staticmethod
    def _uniform_probs(guessed_letters: Set[str]) -> Dict[str, float]:
        """Uniform distribution over unguessed letters."""
        remaining = [chr(ord('a') + i) for i in range(26) 
                     if chr(ord('a') + i) not in guessed_letters]
        if not remaining:
            return {chr(ord('a') + i): 0.0 for i in range(26)}
        
        prob = 1.0 / len(remaining)
        probs = {chr(ord('a') + i): 0.0 for i in range(26)}
        for char in remaining:
            probs[char] = prob
        
        return probs
