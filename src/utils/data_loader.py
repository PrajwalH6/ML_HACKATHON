"""Data loading and preprocessing utilities."""
import re
from collections import defaultdict, Counter
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

class CorpusLoader:
    """Load and preprocess Hangman word corpus."""
    
    def __init__(self, corpus_path='data/raw/corpus.txt'):
        self.corpus_path = Path(corpus_path)
        self.words_by_length = defaultdict(list)
        self.letter_freq = Counter()
        self.position_freq = defaultdict(lambda: defaultdict(int))
        
    def load_and_preprocess(self) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
        """Load corpus, normalize, and organize by length."""
        words = set()
        
        print(f"Loading corpus from {self.corpus_path}...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                word = self._normalize(line.strip())
                if word and self._is_valid(word):
                    words.add(word)
        
        print(f"Loaded {len(words)} unique words")
        
        # Organize by length and compute frequencies
        for word in words:
            length = len(word)
            self.words_by_length[length].append(word)
            
            # Letter frequency
            for char in word:
                self.letter_freq[char] += 1
            
            # Position frequency
            for pos, char in enumerate(word):
                self.position_freq[length][f"{pos}_{char}"] += 1
        
        # Convert to regular dicts
        self.words_by_length = dict(self.words_by_length)
        self.letter_freq = dict(self.letter_freq)
        
        # Print statistics
        print(f"\nWord length distribution:")
        for length in sorted(self.words_by_length.keys()):
            print(f"  Length {length}: {len(self.words_by_length[length])} words")
        
        return self.words_by_length, self.letter_freq
    
    @staticmethod
    def _normalize(word: str) -> str:
        """Convert to lowercase and remove non-alpha."""
        return re.sub(r'[^a-z]', '', word.lower())
    
    @staticmethod
    def _is_valid(word: str) -> bool:
        """Check if word contains only a-z and has length."""
        return bool(word) and word.isalpha() and len(word) > 0
    
    def save_processed(self, output_dir='data/processed'):
        """Save preprocessed data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving preprocessed data to {output_dir}...")
        
        # Save words by length
        with open(output_dir / 'words_by_length.pkl', 'wb') as f:
            pickle.dump(self.words_by_length, f)
        
        # Save letter frequency
        with open(output_dir / 'letter_freq.json', 'w') as f:
            json.dump(self.letter_freq, f, indent=2)
        
        # Save position frequency
        with open(output_dir / 'position_freq.pkl', 'wb') as f:
            pickle.dump(dict(self.position_freq), f)
        
        print("Preprocessing complete!")

    @staticmethod
    def load_processed(data_dir='data/processed') -> Tuple[Dict, Dict]:
        """Load preprocessed data."""
        data_dir = Path(data_dir)
        
        with open(data_dir / 'words_by_length.pkl', 'rb') as f:
            words_by_length = pickle.load(f)
        
        with open(data_dir / 'letter_freq.json', 'r') as f:
            letter_freq = json.load(f)
        
        return words_by_length, letter_freq


def load_test_words(test_path='data/raw/text.txt') -> List[str]:
    """Load test words from text.txt."""
    words = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            word = re.sub(r'[^a-z]', '', line.strip().lower())
            if word:
                words.append(word)
    return words
