"""Training script for RL agent."""
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import json

from src.env.hangman_env import HangmanEnv
from src.hmm.oracle import HMMOracle
from src.hmm.emissions import EmissionBuilder
from src.rl.q_learning import QLearningAgent
from src.utils.data_loader import CorpusLoader
from src.utils.visualization import plot_training_curve
from src.utils.logger import setup_logger

def train_agent(num_episodes=10000, save_interval=1000):
    """Train Q-learning agent."""
    
    # Setup logger
    logger = setup_logger('train_agent', 'models/rl/training.log')
    logger.info("Starting RL agent training...")
    
    # Load preprocessed data
    logger.info("Loading preprocessed data...")
    with open('data/processed/words_by_length.pkl', 'rb') as f:
        words_by_length = pickle.load(f)
    
    # Load HMM emissions
    logger.info("Loading HMM emissions...")
    emissions = EmissionBuilder.load('models/hmm/emissions.pkl')
    
    # Create HMM oracle
    oracle = HMMOracle(emissions, words_by_length)
    
    # Create agent
    agent = QLearningAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.9995,
        epsilon_min=0.01
    )
    
    # Prepare training words
    all_words = []
    for words in words_by_length.values():
        all_words.extend(words)
    
    logger.info(f"Training on {len(all_words)} words for {num_episodes} episodes")
    
    # Training loop
    episode_rewards = []
    episode_wins = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        # Sample random word
        word = random.choice(all_words)
        
        # Create environment
        env = HangmanEnv(word, max_lives=6)
        state = env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            # Get HMM probabilities
            hmm_probs = oracle.get_letter_probs(state['mask'], state['guessed_letters'])
            
            # Select action
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, hmm_probs, valid_actions, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Get next HMM probabilities
            next_hmm_probs = oracle.get_letter_probs(
                next_state['mask'], 
                next_state['guessed_letters']
            )
            
            # Update agent
            agent.update(state, action, reward, next_state, done, hmm_probs, next_hmm_probs)
            
            state = next_state
            total_reward += reward
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Track metrics
        episode_rewards.append(total_reward)
        episode_wins.append(1 if info.get('win', False) else 0)
        
        # Log progress
        if (episode + 1) % 500 == 0:
            recent_wins = sum(episode_wins[-500:])
            recent_reward = sum(episode_rewards[-500:]) / 500
            logger.info(
                f"Episode {episode+1}: "
                f"Win Rate={recent_wins/500:.3f}, "
                f"Avg Reward={recent_reward:.2f}, "
                f"Epsilon={agent.epsilon:.4f}"
            )
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            agent.save(f'models/rl/q_table_ep{episode+1}.pkl')
    
    # Save final model
    agent.save('models/rl/q_table_final.pkl')
    
    # Save training history
    history = {
        'rewards': episode_rewards,
        'wins': episode_wins,
        'num_episodes': num_episodes
    }
    
    Path('models/rl').mkdir(parents=True, exist_ok=True)
    with open('models/rl/training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot training curve
    plot_training_curve(episode_rewards)
    
    logger.info("Training complete!")
    logger.info(f"Final win rate: {sum(episode_wins[-1000:])/1000:.3f}")
    
    return agent

if __name__ == '__main__':
    train_agent(num_episodes=10000)
