"""Experiment 1: which text representation finds the author best?"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from load_data import load

train_texts, train_authors = load("C50train")
test_texts, test_authors = load("C50test")
print("Training articles:", len(train_texts), " Test articles:", len(test_texts))

# keep upper/lower case and punctuation: they are part of an author's style
WORDS = r"\w+|[^\w\s]"

representations = {
    # bag-of-words = plain word counts (use_idf=False)
    "Bag-of-words": TfidfVectorizer(min_df=2, lowercase=False, token_pattern=WORDS, use_idf=False),
    "Word TF-IDF": TfidfVectorizer(min_df=2, lowercase=False, token_pattern=WORDS, sublinear_tf=True),
    "Char n-gram TF-IDF": TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2,
                                          lowercase=False, sublinear_tf=True),
}

for name, vectorizer in representations.items():
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LinearSVC(C=10).fit(x_train, train_authors)
    accuracy = accuracy_score(test_authors, model.predict(x_test))
    print(f"{name}: {100 * accuracy:.1f} % correct")
