#!/bin/bash

echo "Training HMM emissions..."

python -c "
import pickle
from pathlib import Path
from src.hmm.emissions import EmissionBuilder

# Load preprocessed data
with open('data/processed/words_by_length.pkl', 'rb') as f:
    words_by_length = pickle.load(f)

# Build emissions
builder = EmissionBuilder(smoothing_alpha=1.0)
emissions = builder.build_from_corpus(words_by_length)

# Save
Path('models/hmm').mkdir(parents=True, exist_ok=True)
builder.save('models/hmm/emissions.pkl')

print('\nHMM training complete!')
"
