import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer

# - Parser le format .ft.txt
def parse_amazon_review_line(line):
    line = line.strip()
    match = re.match(r"^(__label__\d+)\s+(.*)$", line)
    if match:
        raw_label = match.group(1)
        text = match.group(2)
        return raw_label, text
    return None, None

# chargement du fichier
def load_amazon_ft_file(file_path, max_lines=None):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            raw_label, text = parse_amazon_review_line(line)
            if raw_label and text:
                sentiment = 1 if raw_label == "__label__2" else 0
                rows.append((i, raw_label, sentiment, text))
    
    df = pd.DataFrame(rows, columns=["review_id", "raw_label", "sentiment", "review_text"])
    return df

# Ajout features descriptives

def compute_special_char_ratio(text):
    if not text:
        return 0
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return special_chars / len(text)

def add_text_features(df):
    df = df.copy()
    df["char_len"] = df["review_text"].str.len()
    df["word_len"] = df["review_text"].str.split().str.len()
    df["has_url"] = df["review_text"].str.contains(r"http\S+|www\.\S+", regex=True)
    df["has_html"] = df["review_text"].str.contains(r"<[^>]+>", regex=True)
    df["has_email"] = df["review_text"].str.contains(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", regex=True)
    df["special_char_ratio"] = df["review_text"].apply(compute_special_char_ratio)
    return df

# Création du subset équilibré
def build_balanced_subset(df, n_per_class=3000, random_state=42):
    df_pos = df[df["sentiment"] == 1].sample(n=n_per_class, random_state=random_state)
    df_neg = df[df["sentiment"] == 0].sample(n=n_per_class, random_state=random_state)
    df_subset = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df_subset


# Top 50 mots par classe
def get_top_n(corpus, ngram_range=(1,1), n=20, stop_words='english'):
    vec = CountVectorizer(stop_words=stop_words, ngram_range=ngram_range)
    X = vec.fit_transform(corpus)
    freqs = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    top = sorted(zip(terms, freqs), key=lambda x: x[1], reverse=True)[:n]
    return pd.DataFrame(top, columns=["term", "freq"])