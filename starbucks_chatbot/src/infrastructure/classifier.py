#!/usr/bin/env python3

# src/infrastructure/classifier.py
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from typing import List
import numpy as np
from src.domain.models import Intent
from src.infrastructure.nlp_processor import NLPProcessor

class IntentClassifier:
    def __init__(self, processor: NLPProcessor):
        self.processor = processor
        self.model = Pipeline([
            ('svc', LinearSVC(class_weight='balanced'))
        ])
        self.intent_labels = []

    def train(self, intents: List[Intent]):
        """Entrena el clasificador con las intenciones proporcionadas"""
        X = []  # Características
        y = []  # Etiquetas

        # Preparar datos de entrenamiento
        for intent in intents:
            self.intent_labels.append(intent.name)
            for example in intent.examples:
                processed_text = self.processor.preprocess_text(example)
                features = self.processor.prepare_for_classification(processed_text)
                X.append(features)
                y.append(intent.name)

        # Entrenar modelo
        X = np.vstack(X)
        self.model.fit(X, y)

    def predict(self, text: str) -> str:
        """Predice la intención para un texto dado"""
        processed_text = self.processor.preprocess_text(text)
        features = self.processor.prepare_for_classification(processed_text)
        return self.model.predict(features)[0]