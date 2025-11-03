"""Evaluation script for trained agent."""
import pickle
import json
from pathlib import Path
from tqdm import tqdm

from src.env.hangman_env import HangmanEnv
from src.hmm.oracle import HMMOracle
from src.hmm.emissions import EmissionBuilder
from src.rl.q_learning import QLearningAgent
from src.utils.data_loader import load_test_words
from src.utils.metrics import MetricsCalculator
from src.utils.visualization import plot_evaluation_results
from src.utils.logger import setup_logger

def evaluate_agent(test_words_path='data/raw/text.txt', num_games=2000):
    """Evaluate trained agent on test set."""
    
    # Setup logger
    logger = setup_logger('evaluate', 'reports/results/evaluation.log')
    logger.info("Starting evaluation...")
    
    # Load test words
    logger.info(f"Loading test words from {test_words_path}...")
    all_test_words = load_test_words(test_words_path)
    
    # Sample exactly num_games words (with replacement if needed)
    import random
    if len(all_test_words) >= num_games:
        test_words = random.sample(all_test_words, num_games)
    else:
        # Sample with replacement
        test_words = random.choices(all_test_words, k=num_games)
    
    logger.info(f"Evaluating on {len(test_words)} games")
    
    # Load preprocessed data
    with open('data/processed/words_by_length.pkl', 'rb') as f:
        words_by_length = pickle.load(f)
    
    # Load HMM emissions
    emissions = EmissionBuilder.load('models/hmm/emissions.pkl')
    oracle = HMMOracle(emissions, words_by_length)
    
    # Load trained agent
    agent = QLearningAgent()
    agent.load('models/rl/q_table_final.pkl')
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Metrics calculator
    metrics_calc = MetricsCalculator()
    
    # Evaluate
    for word in tqdm(test_words, desc="Evaluating"):
        env = HangmanEnv(word, max_lives=6)
        state = env.reset()
        done = False
        
        while not done:
            # Get HMM probabilities
            hmm_probs = oracle.get_letter_probs(state['mask'], state['guessed_letters'])
            
            # Select action (no exploration)
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, hmm_probs, valid_actions, training=False)
            
            # Take step
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
    
    # Save metrics
    with open('reports/results/final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    df = metrics_calc.get_dataframe()
    df.to_csv('reports/results/evaluation_results.csv', index=False)
    
    # Plot results
    plot_evaluation_results(metrics, df)
    
    logger.info("Evaluation complete!")
    logger.info("Results saved to reports/results/")
    
    return metrics

if __name__ == '__main__':
    evaluate_agent()
