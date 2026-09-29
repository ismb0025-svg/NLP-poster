"""Read the articles from the data folder."""
import os


def find(name):
    for root, dirs, files in os.walk("data"):
        if name in dirs:
            return os.path.join(root, name)
    raise FileNotFoundError("Run download_data.py first")


def load(split):
    """split is 'C50train' or 'C50test'. Returns the texts and the author names."""
    folder = find(split)
    texts, authors = [], []
    for author in sorted(os.listdir(folder)):
        for filename in sorted(os.listdir(os.path.join(folder, author))):
            path = os.path.join(folder, author, filename)
            with open(path, encoding="utf-8", errors="replace") as f:
                texts.append(f.read())
            authors.append(author)
    return texts, authors
