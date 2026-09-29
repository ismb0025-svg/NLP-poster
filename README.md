# Style, not Semantics?

Code for my NLP poster (University of Trier, 2026).

**Question:** which way of turning text into numbers is best for finding out who
wrote a news article?

**Data:** C50 (Reuter_50_50) – 50 Reuters journalists, 50 training and
50 test articles each.

## How to run

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python download_data.py
python experiment1_representations.py
python experiment2_cleaning.py
python experiment3_less_data.py
```

## Files

- `download_data.py` – downloads the dataset
- `load_data.py` – reads the articles
- `experiment1_representations.py` – bag-of-words vs. TF-IDF vs. character n-grams
- `experiment2_cleaning.py` – does cleaning the text help?
- `experiment3_less_data.py` – fewer training articles per author
- `experiment4_word_vectors.py` – GloVe word vectors
- `experiment5_important_words.py` – which words matter most for each author

## Results

| Representation | Correct authors |
|---|---|
| Char n-gram TF-IDF | 76.2 % |
| Word TF-IDF | 73.9 % |
| Bag-of-words | 72.9 % |
| GloVe (word vectors) | 63.4 % |

Cleaning the text never helped. Character n-grams also win when only a few
training articles per author are available.
