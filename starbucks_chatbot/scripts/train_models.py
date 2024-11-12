#!/usr/bin/env python3
# scripts/train_models.py

import json
from pathlib import Path
from src.application.training.train_models_usecase import (
    TrainModelsUseCase,
    TrainingConfig
)
from src.infrastructure.training.data_preparation import DataPreparation
from src.infrastructure.training.vectorizer_trainer import VectorizerTrainer
from src.infrastructure.persistence.file_model_storage import FileModelStorage
from src.infrastructure.logging.training_logger import ConsoleTrainingLogger

def load_training_data(data_path: Path) -> TrainingConfig:
    """Carga la configuración de entrenamiento"""
    with open(data_path / 'menu.json', 'r', encoding='utf-8') as f:
        menu_data = json.load(f)

    with open(data_path / 'intents.json', 'r', encoding='utf-8') as f:
        intent_data = json.load(f)

    return TrainingConfig(
        menu_data=menu_data,
        intent_data=intent_data,
        model_path=str(data_path.parent / 'models' / 'vectorizer.joblib')
    )

def main():
    # Configurar componentes
    data_prep = DataPreparation()
    vectorizer_trainer = VectorizerTrainer()
    model_storage = FileModelStorage()
    logger = ConsoleTrainingLogger()

    # Crear caso de uso
    use_case = TrainModelsUseCase(
        data_preparation=data_prep,
        vectorizer_trainer=vectorizer_trainer,
        model_storage=model_storage,
        logger=logger
    )

    # Cargar configuración
    config = load_training_data(Path(__file__).parent.parent / 'data')

    # Ejecutar entrenamiento
    try:
        model_path = use_case.execute(config)
        print(f"Entrenamiento completado exitosamente: {model_path}")
        return 0
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return 1

if __name__ == "__main__":
    exit(main())