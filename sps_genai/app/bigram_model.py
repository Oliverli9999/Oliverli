# app/main.py
import random
from collections import defaultdict
from fastapi import FastAPI, HTTPException
import spacy

# ---------------- Bigram Model ----------------
class BigramModel:
    def __init__(self, corpus):
        """
        Initialize the Bigram model.
        corpus: list of sentences used for training.
        """
        self.model = defaultdict(list)
        self.train(corpus)

    def train(self, corpus):
        """
        Train the bigram model by creating a mapping:
        word -> list of possible next words.
        """
        for sentence in corpus:
            words = sentence.lower().split()
            for i in range(len(words) - 1):
                self.model[words[i]].append(words[i + 1])

    def generate_text(self, start_word, length=10):
        """
        Generate text of given length starting from 'start_word'.
        At each step, randomly choose a next word from the bigram mapping.
        """
        word = start_word.lower()
        result = [word]
        for _ in range(length - 1):
            if word not in self.model:
                break
            word = random.choice(self.model[word])
            result.append(word)
        return " ".join(result)


# ---------------- FastAPI ----------------
app = FastAPI(title="Bigram + Embedding API")

# Example training corpus for the Bigram model
corpus = [
    "I love New York",
    "I love pizza",
    "New York loves pizza",
    "Pizza in New York is great"
]
bigram_model = BigramModel(corpus)

# Load spaCy model that contains word vectors
# (use en_core_web_md or en_core_web_lg, not en_core_web_sm)
nlp = spacy.load("en_core_web_md")


@app.get("/")
def root():
    """Root endpoint to check if the API is running."""
    return {"message": "Bigram + Embedding API running"}


@app.get("/generate")
def generate(start_word: str, length: int = 10):
    """
    Generate text using the Bigram model.
    Params:
      - start_word: the word to begin generation
      - length: maximum number of words to generate
    """
    text = bigram_model.generate_text(start_word, length)
    return {"start": start_word, "generated": text}


@app.get("/embedding")
def embedding(word: str):
    """
    Return the spaCy embedding vector for a given word or phrase.
    By default, spaCy averages token vectors if the input has multiple tokens.
    """
    doc = nlp(word)
    if not doc:
        raise HTTPException(status_code=400, detail="Empty input")
    vec = doc.vector  # average vector for whole phrase
    # Truncate output to first 10 dimensions to avoid huge JSON
    return {"word": word, "embedding": vec.tolist()[:10]}
