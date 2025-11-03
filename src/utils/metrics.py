"""Evaluation metrics calculator."""
from typing import List, Dict
import pandas as pd

class MetricsCalculator:
    """Calculate Hangman evaluation metrics."""
    
    def __init__(self):
        self.results = []
    
    def add_game(self, word: str, won: bool, wrong_guesses: int, repeated_guesses: int):
        """Add a game result."""
        self.results.append({
            'word': word,
            'won': won,
            'wrong_guesses': wrong_guesses,
            'repeated_guesses': repeated_guesses
        })
    
    def compute_metrics(self) -> Dict:
        """Compute final metrics."""
        df = pd.DataFrame(self.results)
        
        total_games = len(df)
        wins = df['won'].sum()
        success_rate = wins / total_games if total_games > 0 else 0
        
        total_wrong = df['wrong_guesses'].sum()
        total_repeated = df['repeated_guesses'].sum()
        
        # Final Score formula from problem statement
        final_score = (success_rate * total_games) - (total_wrong * 5) - (total_repeated * 2)
        
        metrics = {
            'total_games': total_games,
            'wins': int(wins),
            'losses': int(total_games - wins),
            'success_rate': success_rate,
            'total_wrong_guesses': int(total_wrong),
            'total_repeated_guesses': int(total_repeated),
            'avg_wrong_per_game': total_wrong / total_games if total_games > 0 else 0,
            'avg_repeated_per_game': total_repeated / total_games if total_games > 0 else 0,
            'final_score': final_score
        }
        
        return metrics
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return results as DataFrame."""
        return pd.DataFrame(self.results)
