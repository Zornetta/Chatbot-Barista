#!/usr/bin/env python3

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import joblib

from src.domain.models import Intent
from src.infrastructure.nlp_processor import NLPProcessor

@dataclass
class ClassificationMetrics:
    """Métricas de evaluación del clasificador"""
    accuracy: float
    report: Dict
    confusion_matrix: Optional[np.ndarray] = None

class IntentClassifier:
    def __init__(self, processor: NLPProcessor, model_path: str = None):
        """
        Inicializa el clasificador de intenciones

        Args:
            processor: NLPProcessor inicializado
            model_path: Ruta opcional al modelo guardado
        """
        self.processor = processor
        self.model = self._create_pipeline()
        self.intent_labels: List[str] = []
        self._load_model(model_path) if model_path else None

    def _create_pipeline(self) -> Pipeline:
        """Crea el pipeline de clasificación"""
        return Pipeline([
            ('svc', LinearSVC(
                class_weight='balanced',
                C=1.0,
                max_iter=1000,
                random_state=42
            ))
        ])

    def _load_model(self, path: str) -> None:
        """
        Carga un modelo guardado

        Args:
            path: Ruta al modelo

        Raises:
            RuntimeError: Si hay error al cargar el modelo
        """
        try:
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.intent_labels = model_data['labels']
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo: {str(e)}")

    def save_model(self, path: str) -> None:
        """
        Guarda el modelo entrenado

        Args:
            path: Ruta donde guardar el modelo

        Raises:
            RuntimeError: Si hay error al guardar el modelo
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'labels': self.intent_labels
            }
            joblib.dump(model_data, path)
        except Exception as e:
            raise RuntimeError(f"Error al guardar el modelo: {str(e)}")

    def train(self, intents: List[Intent], test_size: float = 0.2) -> ClassificationMetrics:
        """
        Entrena el clasificador con las intenciones proporcionadas

        Args:
            intents: Lista de intenciones para entrenamiento
            test_size: Proporción de datos para testing

        Returns:
            Métricas de evaluación del modelo

        Raises:
            ValueError: Si no hay suficientes datos de entrenamiento
        """
        if not intents:
            raise ValueError("No hay datos de entrenamiento")

        try:
            # Preparar datos
            X, y = self._prepare_training_data(intents)

            if len(np.unique(y)) < 2:
                raise ValueError("Se necesitan al menos 2 clases diferentes para entrenar")

            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Entrenar modelo
            self.model.fit(X_train, y_train)

            # Evaluar
            y_pred = self.model.predict(X_test)
            metrics = ClassificationMetrics(
                accuracy=self.model.score(X_test, y_test),
                report=classification_report(y_test, y_pred, output_dict=True)
            )

            return metrics

        except Exception as e:
            raise RuntimeError(f"Error durante el entrenamiento: {str(e)}")

    def _prepare_training_data(self, intents: List[Intent]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara los datos de entrenamiento

        Args:
            intents: Lista de intenciones

        Returns:
            Tupla de (features, labels)
        """
        processed_texts = []  # Lista para textos procesados
        labels = []  # Lista para etiquetas
        self.intent_labels = []

        # Preparar datos
        for intent in intents:
            self.intent_labels.append(intent.name)
            for example in intent.examples:
                processed_text = self.processor.preprocess_text(example)
                processed_texts.append(processed_text)
                labels.append(intent.name)

        # Vectorizar todos los textos de una vez
        X = self.processor.vectorizer.transform(processed_texts)
        y = np.array(labels)

        return X, y

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predice la intención para un texto dado

        Args:
            text: Texto de entrada

        Returns:
            Tupla de (intención predicha, confianza)

        Raises:
            RuntimeError: Si el modelo no está entrenado
        """
        if not self.intent_labels:
            raise RuntimeError("El modelo no está entrenado")

        try:
            # Preprocesar y vectorizar
            processed_text = self.processor.preprocess_text(text)
            features = self.processor.prepare_for_classification(processed_text)

            # Predecir
            intent = self.model.predict(features)[0]

            # Obtener confianza (distancia al hiperplano)
            confidence = float(abs(self.model.decision_function(features)[0]))

            return intent, confidence

        except Exception as e:
            raise RuntimeError(f"Error en la predicción: {str(e)}")

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Predice intenciones para una lista de textos

        Args:
            texts: Lista de textos

        Returns:
            Lista de tuplas (intención, confianza)
        """
        return [self.predict(text) for text in texts]

    def get_supported_intents(self) -> List[str]:
        """
        Obtiene las intenciones soportadas por el modelo

        Returns:
            Lista de nombres de intenciones
        """
        return self.intent_labels.copy()