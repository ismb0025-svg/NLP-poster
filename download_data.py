"""Download the C50 dataset (50 Reuters journalists, 100 articles each)."""
import urllib.request, zipfile, os

URL = "https://archive.ics.uci.edu/static/public/217/reuter+50+50.zip"

if not os.path.exists("data"):
    print("Downloading the dataset...")
    urllib.request.urlretrieve(URL, "c50.zip")
    with zipfile.ZipFile("c50.zip") as z:
        z.extractall("data")
    print("Done. The data is in the folder 'data'.")
else:
    print("The folder 'data' already exists.")
