"""Experiment 4: GloVe word vectors (the 'meaning' method) instead of counting."""
import gzip
import os
import re
import urllib.request

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from load_data import load

URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
       "glove-wiki-gigaword-300/glove-wiki-gigaword-300.gz")

if not os.path.exists("glove.gz"):
    print("Downloading GloVe word vectors (about 400 MB)...")
    urllib.request.urlretrieve(URL, "glove.gz")

print("Reading the word vectors...")
vectors = {}
with gzip.open("glove.gz", "rt", encoding="utf-8") as f:
    next(f)  # first line only says how many words and how long the vectors are
    for line in f:
        parts = line.split(" ")
        vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)


def to_vectors(texts):
    """One article = the average of the vectors of its words."""
    out = np.zeros((len(texts), 300), dtype=np.float32)
    for i, text in enumerate(texts):
        found = [vectors[w] for w in re.findall(r"\w+", text.lower()) if w in vectors]
        if found:
            out[i] = np.mean(found, axis=0)
    return out


train_texts, train_authors = load("C50train")
test_texts, test_authors = load("C50test")
x_train, x_test = to_vectors(train_texts), to_vectors(test_texts)
scaler = StandardScaler().fit(x_train)
x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
train_authors, test_authors = np.array(train_authors), np.array(test_authors)

model = LinearSVC(C=0.01).fit(x_train, train_authors)
print(f"GloVe, all 50 articles per author: {100 * accuracy_score(test_authors, model.predict(x_test)):.1f} %")

for n in [5, 10, 20]:
    rng = np.random.default_rng(0)
    chosen = np.concatenate([rng.permutation(np.where(train_authors == a)[0])[:n]
                             for a in np.unique(train_authors)])
    model = LinearSVC(C=0.01).fit(x_train[chosen], train_authors[chosen])
    print(f"GloVe, {n} articles per author: {100 * accuracy_score(test_authors, model.predict(x_test)):.1f} %")
