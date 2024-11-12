# src/infrastructure/training/vectorizer_trainer.py
import spacy
from src.domain.training.interfaces import IVectorizerTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Any

class VectorizerTrainer(IVectorizerTrainer):
    def __init__(self):
        # Cargar spaCy para obtener stop words en español
        self.nlp = spacy.load("es_core_news_sm")
        # Obtener lista de stop words
        self.stop_words = list(self.nlp.Defaults.stop_words)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words=self.stop_words  # Usar lista de stop words de spaCy
        )

    def train(self, texts: List[str]) -> Any:
        if not texts:
            raise ValueError("No se pueden entrenar con textos vacíos")

        self.vectorizer.fit(texts)
        return self.vectorizer