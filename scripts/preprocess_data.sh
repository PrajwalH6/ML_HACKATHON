#!/bin/bash

echo "Preprocessing corpus data..."

python -c "
from src.utils.data_loader import CorpusLoader

loader = CorpusLoader('data/raw/corpus.txt')
words_by_length, letter_freq = loader.load_and_preprocess()
loader.save_processed()

print('\nPreprocessing complete!')
"
