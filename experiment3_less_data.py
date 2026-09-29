"""Experiment 3: what if we have only a few articles per author?"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from load_data import load

train_texts, train_authors = load("C50train")
test_texts, test_authors = load("C50test")
train_texts, train_authors = np.array(train_texts), np.array(train_authors)

methods = {
    "Char n-gram TF-IDF": lambda: TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                                  lowercase=False, sublinear_tf=True),
    "Word TF-IDF": lambda: TfidfVectorizer(min_df=2, lowercase=False,
                                           token_pattern=r"\w+|[^\w\s]", sublinear_tf=True),
}

for name, make_vectorizer in methods.items():
  for n in [5, 10, 20, 50]:
      rng = np.random.default_rng(0)
      chosen = np.concatenate([rng.permutation(np.where(train_authors == a)[0])[:n]
                               for a in np.unique(train_authors)])
      vectorizer = make_vectorizer()
      x_train = vectorizer.fit_transform(train_texts[chosen])
      x_test = vectorizer.transform(test_texts)
      model = LinearSVC(C=10).fit(x_train, train_authors[chosen])
      accuracy = accuracy_score(test_authors, model.predict(x_test))
      print(f"{name}, {n} articles per author: {100 * accuracy:.1f} %")
