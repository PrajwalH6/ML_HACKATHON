"""Enhanced evaluation script using Pure HMM + Smart Strategy."""
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from src.env.hangman_env import HangmanEnv
from src.hmm.oracle import HMMOracle
from src.hmm.emissions import EmissionBuilder
from src.utils.data_loader import load_test_words
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import plot_evaluation_results
from src.utils.logger import setup_logger


class SmartHangmanAgent:
    """Enhanced agent using HMM + frequency + vowel heuristics."""
    
    def __init__(self, oracle, letter_freq):
        self.oracle = oracle
        # Normalize letter frequencies
        total = sum(letter_freq.values())
        self.letter_probs = {k: v/total for k, v in letter_freq.items()}
        self.vowels = set('aeiou')
        self.common_consonants = 'tnshrdlcmwfgypbvkjxqz'
    
    def select_action(self, state, valid_actions):
        """Smart action selection combining multiple strategies."""
        mask = state['mask']
        guessed = state['guessed_letters']
        
        # Get HMM predictions
        hmm_probs = self.oracle.get_letter_probs(mask, guessed)
        
        # Strategy 1: Trust HMM if confident (>5% probability)
        valid_hmm = {l: hmm_probs.get(l, 0.0) for l in valid_actions}
        max_hmm_prob = max(valid_hmm.values()) if valid_hmm else 0
        
        if max_hmm_prob > 0.05:
            return max(valid_hmm, key=valid_hmm.get)
        
        # Strategy 2: Early game - prioritize vowels
        blanks_ratio = mask.count('_') / len(mask)
        if blanks_ratio > 0.7:  # >70% unknown
            available_vowels = self.vowels & valid_actions
            if available_vowels:
                # Pick most frequent available vowel
                return max(available_vowels, 
                          key=lambda l: self.letter_probs.get(l, 0))
        
        # Strategy 3: Mid/late game - use corpus frequency
        valid_freq = {l: self.letter_probs.get(l, 0) for l in valid_actions}
        if valid_freq and sum(valid_freq.values()) > 0:
            # Boost common consonants
            for letter in valid_freq:
                if letter in self.common_consonants[:10]:
                    valid_freq[letter] *= 1.5
            
            return max(valid_freq, key=valid_freq.get)
        
        # Strategy 4: Fallback to frequency order
        for letter in 'etaoinshrdlcumwfgypbvkjxqz':
            if letter in valid_actions:
                return letter
        
        # Last resort
        return list(valid_actions)[0] if valid_actions else 'e'


def evaluate_agent(test_words_path='data/raw/text.txt', num_games=2000):
    """Evaluate enhanced agent on test set."""
    
    # Setup logger
    logger = setup_logger('evaluate', 'reports/results/evaluation.log')
    logger.info("Starting evaluation with Enhanced HMM Strategy...")
    
    # Load test words
    logger.info(f"Loading test words from {test_words_path}...")
    all_test_words = load_test_words(test_words_path)
    
    # Sample exactly num_games words
    import random
    if len(all_test_words) >= num_games:
        test_words = random.sample(all_test_words, num_games)
    else:
        test_words = random.choices(all_test_words, k=num_games)
    
    logger.info(f"Evaluating on {len(test_words)} games")
    
    # Load preprocessed data
    with open('data/processed/words_by_length.pkl', 'rb') as f:
        words_by_length = pickle.load(f)
    
    with open('data/processed/letter_freq.json', 'r') as f:
        letter_freq = json.load(f)
    
    # Load HMM emissions
    emissions = EmissionBuilder.load('models/hmm/emissions.pkl')
    oracle = HMMOracle(emissions, words_by_length)
    
    # Create enhanced agent
    agent = SmartHangmanAgent(oracle, letter_freq)
    logger.info("Using Enhanced HMM + Frequency + Vowel Strategy")
    
    # Metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Evaluate
    for word in tqdm(test_words, desc="Evaluating"):
        env = HangmanEnv(word, max_lives=6)
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            state, reward, done, info = env.step(action)
        
        # Record result
        metrics_calc.add_game(
            word=word,
            won=info.get('win', False),
            wrong_guesses=env.wrong_guesses,
            repeated_guesses=env.repeated_guesses
        )
    
    # Compute metrics
    metrics = metrics_calc.compute_metrics()
    
    # Log results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Total Games: {metrics['total_games']}")
    logger.info(f"Wins: {metrics['wins']}")
    logger.info(f"Losses: {metrics['losses']}")
    logger.info(f"Success Rate: {metrics['success_rate']*100:.2f}%")
    logger.info(f"Total Wrong Guesses: {metrics['total_wrong_guesses']}")
    logger.info(f"Total Repeated Guesses: {metrics['total_repeated_guesses']}")
    logger.info(f"Avg Wrong/Game: {metrics['avg_wrong_per_game']:.2f}")
    logger.info(f"Avg Repeated/Game: {metrics['avg_repeated_per_game']:.2f}")
    logger.info(f"\nFINAL SCORE: {metrics['final_score']:.2f}")
    logger.info("="*50)
    
    # Save results
    Path('reports/results').mkdir(parents=True, exist_ok=True)
    
    with open('reports/results/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    df = metrics_calc.get_dataframe()
    df.to_csv('reports/results/evaluation_results.csv', index=False)
    
    # Plot results
    plot_evaluation_results(metrics, df)
    
    logger.info("Evaluation complete!")
    logger.info("Results saved to reports/results/")
    
    return metrics


if __name__ == '__main__':
    evaluate_agent()
