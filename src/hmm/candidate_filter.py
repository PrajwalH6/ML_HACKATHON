"""Fast candidate word filtering."""
from typing import List, Set

def filter_candidates(words: List[str], mask: str, guessed: Set[str]) -> List[str]:
    """Filter words matching mask and guessed constraints."""
    candidates = []
    
    for word in words:
        if matches_mask(word, mask, guessed):
            candidates.append(word)
    
    return candidates


def matches_mask(word: str, mask: str, guessed: Set[str]) -> bool:
    """Check if word matches current mask and constraints."""
    if len(word) != len(mask):
        return False
    
    for w_char, m_char in zip(word, mask):
        if m_char != '_' and w_char != m_char:
            return False
        if m_char == '_' and w_char in guessed:
            return False
    
    # Check word doesn't contain wrong guesses
    for char in guessed:
        if char not in mask and char in word:
            return False
    
    return True
