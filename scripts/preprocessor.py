import re
import unicodedata
import html

class TextPreprocessor:

    def __init__(self, nlp=None, stop_words:list=None, remove_accents:bool=True, remove_numbers:bool=True, use_lemmatization:bool=False):
        self.nlp = nlp
        self.stop_words = set(stop_words) if stop_words is not None else set()
        self.remove_accents = remove_accents
        self.remove_numbers = remove_numbers
        self.use_lemmatization = use_lemmatization

    def process(self, text:str):
        text = self._clean_raw(text)
        text = self._remove_html(text)
        text = self._normalize(text)
        tokens = self._tokenize(text)
        tokens = self._remove_stopwords(tokens)

        if self.use_lemmatization:
            tokens = self._lemmatize(tokens)

        return tokens

    def _clean_raw(self, text:list[str]):
        """
        Raw cleaning:
        - Remove reviews that are too short (< 5 characters)
        - Remove reviews with too many special characters (>95%)
        - Handle UTF-8 encoding
        """

        if text is None:
            return ""

        # Ensure UTF-8 encoding
        text = str(text).encode("utf-8", errors="ignore").decode("utf-8")
        text = text.strip()

        # Remove reviews that are too short
        if len(text) < 5:
            return ""

        # Compute special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if len(text) > 0 and (special_chars / len(text)) > 0.95:
            return ""

        return text

    def _remove_html(self, text:list[str]):
        """
        Remove HTML, URLs, mentions, and decode HTML entities.
        """
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove user mentions (@username)
        text = re.sub(r"@\w+", "", text)

        return text

    def _normalize(self, text:list[str]):
        text = text.lower()

        # Remove accents
        if self.remove_accents:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(c for c in text if not unicodedata.combining(c))

        # Handle numbers
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _tokenize(self, text:list[str]):
        if not text:
            return []
        return text.split()

    def _remove_stopwords(self, tokens:list[str]):
        return [token for token in tokens if token not in self.stop_words]

    def _lemmatize(self, tokens:list[str]):
        if not tokens:
            return []

        if self.nlp is None:
            raise ValueError("The spaCy 'nlp' model must be provided to use lemmatization.")

        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc]
