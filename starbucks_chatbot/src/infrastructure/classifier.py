#!/usr/bin/env python3

import time
from functools import wraps
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from typing import List, Dict, Tuple, Optional, Callable, TypeVar, Any
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import joblib

from src.domain.models import Intent
from src.infrastructure.nlp_processor import NLPProcessor

T = TypeVar('T')

def retry_with_backoff(retries: int = 3, backoff_in_seconds: int = 1):
    """
    Decorador que implementa reintentos con backoff exponencial

    Args:
        retries: Número de reintentos
        backoff_in_seconds: Tiempo base de espera entre reintentos

    Returns:
        Wrapper function
    """
    def decorator(operation: Callable[..., T]) -> Callable[..., T]:
        @wraps(operation)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(retries + 1):  # +1 para incluir el intento original
                try:
                    return operation(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == retries:  # Si es el último intento, propagar el error
                        raise RuntimeError(
                            f"Error después de {retries + 1} intentos. "
                            f"Último error: {str(last_exception)}"
                        )
                    # Calcular tiempo de espera exponencial
                    wait_time = backoff_in_seconds * (2 ** attempt)
                    time.sleep(wait_time)
            raise last_exception  # Por si acaso, nunca deberíamos llegar aquí
        return wrapper
    return decorator

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

        # Cargar modelo si se proporciona ruta
        if model_path:
            self._load_model(model_path)
        else:
            # Buscar modelo en ubicación por defecto
            try:
                default_path = Path(__file__).parent.parent.parent / "models" / "classifier.joblib"
                if default_path.exists():
                    self._load_model(str(default_path))
            except Exception:
                # Si no se encuentra el modelo por defecto, se iniciará sin modelo
                pass

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

    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def _load_model(self, path: str) -> None:
        """
        Carga un modelo guardado con reintentos

        Args:
            path: Ruta al modelo

        Raises:
            RuntimeError: Si hay error al cargar el modelo después de todos los reintentos
        """
        try:
            model_data = joblib.load(path)
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.intent_labels = model_data.get('labels', [])
            else:
                # Compatibilidad con versiones anteriores donde solo se guardaba el modelo
                self.model = model_data
                # Intentar inferir las etiquetas desde el modelo
                if hasattr(self.model, 'classes_'):
                    self.intent_labels = list(self.model.classes_)
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo desde {path}: {str(e)}")

    def save_model(self, path: str) -> None:
        """
        Guarda el modelo entrenado

        Args:
            path: Ruta donde guardar el modelo

        Raises:
            RuntimeError: Si hay error al guardar el modelo
        """
        if not self.model or not self.intent_labels:
            raise RuntimeError("No hay modelo entrenado para guardar")

        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            model_data = {
                'model': self.model,
                'labels': self.intent_labels
            }
            joblib.dump(model_data, path)
        except Exception as e:
            raise RuntimeError(f"Error al guardar el modelo en {path}: {str(e)}")

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

            # Actualizar intent_labels después del entrenamiento
            if hasattr(self.model, 'classes_'):
                self.intent_labels = list(self.model.classes_)

            # Evaluar
            y_pred = self.model.predict(X_test)
            metrics = ClassificationMetrics(
                accuracy=self.model.score(X_test, y_test),
                report=classification_report(y_test, y_pred, output_dict=True)
            )

            return metrics

        except Exception as e:
            raise RuntimeError(f"Error durante el entrenamiento: {str(e)}")

    @retry_with_backoff(retries=3, backoff_in_seconds=1)
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predice la intención para un texto dado con reintentos

        Args:
            text: Texto de entrada

        Returns:
            Tupla de (intención predicha, confianza)

        Raises:
            RuntimeError: Si el modelo no está entrenado o hay error en la predicción
            después de todos los reintentos
        """
        if not hasattr(self.model, 'predict') or not self.intent_labels:
            raise RuntimeError("El modelo no está entrenado o no se ha cargado correctamente")

        try:
            # Preprocesar y vectorizar
            processed_text = self.processor.preprocess_text(text)
            features = self.processor.prepare_for_classification(processed_text)

            # Predecir
            intent = self.model.predict(features)[0]

            # Obtener confianza (distancia al hiperplano)
            decision_values = self.model.decision_function(features)
            confidence = float(np.abs(decision_values).max())

            return intent, confidence

        except Exception as e:
            raise RuntimeError(f"Error en la predicción: {str(e)}")

    def _prepare_training_data(self, intents: List[Intent]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara los datos de entrenamiento

        Args:
            intents: Lista de intenciones

        Returns:
            Tupla de (features, labels)
        """
        processed_texts = []
        labels = []
        self.intent_labels = []

        # Preparar datos
        for intent in intents:
            self.intent_labels.append(intent.name)
            for example in intent.examples:
                processed_text = self.processor.preprocess_text(example)
                processed_texts.append(processed_text)
                labels.append(intent.name)

        # Vectorizar textos
        X = self.processor.vectorizer.transform(processed_texts)
        y = np.array(labels)

        return X, y

    def get_supported_intents(self) -> List[str]:
        """Obtiene las intenciones soportadas por el modelo"""
        return self.intent_labels.copy()