# src/application/training/train_models_usecase.py
from dataclasses import dataclass
from typing import Dict, List
from src.domain.training.interfaces import (
    IDataPreparation,
    IVectorizerTrainer,
    IModelStorage,
    ITrainingLogger
)

@dataclass
class TrainingConfig:
    """Configuración del entrenamiento"""
    menu_data: Dict
    intent_data: List
    model_path: str

class TrainModelsUseCase:
    def __init__(
        self,
        data_preparation: IDataPreparation,
        vectorizer_trainer: IVectorizerTrainer,
        model_storage: IModelStorage,
        logger: ITrainingLogger
    ):
        self.data_preparation = data_preparation
        self.vectorizer_trainer = vectorizer_trainer
        self.model_storage = model_storage
        self.logger = logger

    def execute(self, config: TrainingConfig) -> str:
        """
        Ejecuta el proceso de entrenamiento completo

        Returns:
            str: Ruta del modelo guardado
        """
        self.logger.info("Iniciando proceso de entrenamiento")

        try:
            # Preparar datos
            menu_corpus = self.data_preparation.prepare_menu_data(config.menu_data)
            intent_corpus = self.data_preparation.prepare_intent_data(config.intent_data)

            # Combinar datasets
            self.logger.info("Combinando datasets")
            combined_corpus = self.data_preparation.combine_datasets(
                menu_corpus,
                intent_corpus
            )

            # Entrenar vectorizador
            self.logger.info("Entrenando vectorizador")
            vectorizer = self.vectorizer_trainer.train(combined_corpus)

            # Guardar modelo
            self.logger.info("Guardando modelo entrenado")
            model_path = self.model_storage.save_vectorizer(
                vectorizer,
                path=config.model_path
            )

            self.logger.info(f"Entrenamiento completado: {model_path}")
            return model_path

        except Exception as e:
            self.logger.error(f"Error en entrenamiento: {str(e)}")
            raise