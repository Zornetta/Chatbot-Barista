# src/domain/training/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

class IDataPreparation(ABC):
    @abstractmethod
    def prepare_menu_data(self, menu_data: Dict) -> List[str]:
        """
        Prepara los datos del menú para entrenamiento.

        Args:
            menu_data: Diccionario con datos del menú

        Returns:
            Lista de textos procesados del menú
        """
        pass

    @abstractmethod
    def prepare_intent_data(self, intent_data: List) -> List[str]:
        """
        Prepara los datos de intenciones para entrenamiento.

        Args:
            intent_data: Lista de intenciones y ejemplos

        Returns:
            Lista de textos procesados de intenciones
        """
        pass

    @abstractmethod
    def combine_datasets(self, menu_corpus: List[str], intent_corpus: List[str]) -> List[str]:
        """
        Combina los datasets preparados.

        Args:
            menu_corpus: Textos procesados del menú
            intent_corpus: Textos procesados de intenciones

        Returns:
            Dataset combinado y normalizado
        """
        pass

class IVectorizerTrainer(ABC):
    @abstractmethod
    def train(self, texts: List[str]) -> Any:
        """
        Entrena el vectorizador con los textos proporcionados.

        Args:
            texts: Lista de textos para entrenamiento

        Returns:
            Vectorizador entrenado
        """
        pass

class IModelStorage(ABC):
    @abstractmethod
    def save_vectorizer(self, vectorizer: Any, path: str) -> str:
        """
        Guarda el vectorizador entrenado.

        Args:
            vectorizer: Vectorizador entrenado
            path: Ruta donde guardar el modelo

        Returns:
            Ruta donde se guardó el modelo
        """
        pass

    @abstractmethod
    def load_vectorizer(self, path: str) -> Any:
        """
        Carga un vectorizador entrenado.

        Args:
            path: Ruta del modelo a cargar

        Returns:
            Vectorizador cargado
        """
        pass

class ITrainingLogger(ABC):
    @abstractmethod
    def info(self, message: str) -> None:
        """Registra mensaje informativo"""
        pass

    @abstractmethod
    def error(self, message: str) -> None:
        """Registra mensaje de error"""
        pass