# src/application/training/training_service.py
from src.domain.training.interfaces import IDataPreparation, IVectorizerTrainer
from src.infrastructure.persistence.model_storage import ModelStorage
from typing import Dict, List

class TrainingService:
    def __init__(
        self,
        data_preparation: IDataPreparation,
        vectorizer_trainer: IVectorizerTrainer,
        model_storage: ModelStorage
    ):
        self.data_preparation = data_preparation
        self.vectorizer_trainer = vectorizer_trainer
        self.model_storage = model_storage

    def train_vectorizer(self, menu_data: Dict, intent_data: List) -> str:
        """Entrena el vectorizador completo"""
        # Preparar datos
        menu_corpus = self.data_preparation.prepare_menu_data(menu_data)
        intent_corpus = self.data_preparation.prepare_intent_data(intent_data)

        # Combinar datasets
        combined_corpus = self.data_preparation.combine_datasets(
            menu_corpus,
            intent_corpus
        )

        # Entrenar vectorizador
        vectorizer = self.vectorizer_trainer.train(combined_corpus)

        # Guardar modelo
        model_path = self.model_storage.save_vectorizer(vectorizer)

        return model_path