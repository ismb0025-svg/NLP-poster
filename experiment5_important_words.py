"""Experiment 5: which words are most important for recognising an author?"""
import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC

from load_data import load

train_texts, train_authors = load("C50train")
vectorizer = TfidfVectorizer(min_df=2, lowercase=False, token_pattern=r"\w+|[^\w\s]",
                             sublinear_tf=True)
x_train = vectorizer.fit_transform(train_texts)
model = LinearSVC(C=10).fit(x_train, train_authors)
words = np.array(vectorizer.get_feature_names_out())


def kind(word):
    if not re.search(r"\w", word):
        return "punctuation"
    if word.lower() in ENGLISH_STOP_WORDS:
        return "function word"
    if any(c.isdigit() for c in word):
        return "number"
    if word[0].isupper():
        return "name (capitalised)"
    return "other word"


top = np.argsort(-model.coef_, axis=1)[:, :30]          # 30 strongest words per author
kinds_top = np.array([[kind(w) for w in words[row]] for row in top])
kinds_all = np.array([kind(w) for w in words])

print("Examples (first 3 authors):")
for i in range(3):
    print(" ", model.classes_[i], ":", ", ".join(words[top[i]][:8]))

print("\nShare among the 30 strongest words per author:")
for k in ["name (capitalised)", "other word", "function word", "number", "punctuation"]:
    print(f"  {k:>18}: {100 * np.mean(kinds_top == k):.1f} %   "
          f"(in the whole vocabulary: {100 * np.mean(kinds_all == k):.1f} %)")
