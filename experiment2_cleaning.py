"""Experiment 2: does cleaning the text help? (stop words, stemming, lemmas)"""
import re

import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from load_data import load

train_texts, train_authors = load("C50train")
test_texts, test_authors = load("C50test")
stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def clean(texts, how):
    if how == "raw":
        return texts
    if how == "lowercase":
        return [t.lower() for t in texts]
    if how == "no stop words":
        return [" ".join(w for w in re.findall(r"\w+|[^\w\s]", t) if w.lower() not in ENGLISH_STOP_WORDS)
                for t in texts]
    if how == "stemmed":
        return [" ".join(stemmer.stem(w) for w in re.findall(r"\w+|[^\w\s]", t)) for t in texts]
    if how == "lemmatized":
        return [" ".join(token.lemma_ for token in doc) for doc in nlp.pipe(texts)]


for how in ["raw", "lowercase", "no stop words", "stemmed", "lemmatized"]:
    train_clean, test_clean = clean(train_texts, how), clean(test_texts, how)
    for name, vectorizer in [("Word TF-IDF", TfidfVectorizer(min_df=2, lowercase=False,
                                                       token_pattern=r"\w+|[^\w\s]", sublinear_tf=True)),
                             ("Char n-gram TF-IDF", TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                                                     min_df=2, lowercase=False, sublinear_tf=True))]:
        x_train = vectorizer.fit_transform(train_clean)
        x_test = vectorizer.transform(test_clean)
        model = LinearSVC(C=10).fit(x_train, train_authors)
        accuracy = accuracy_score(test_authors, model.predict(x_test))
        print(f"{how:>14} | {name:>18}: {100 * accuracy:.1f} %")
