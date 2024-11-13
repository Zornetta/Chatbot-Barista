#!/usr/bin/env python3
# scripts/train_classifier.py

import json
from pathlib import Path
from src.infrastructure.nlp_processor import NLPProcessor
from src.infrastructure.classifier import IntentClassifier
from src.domain.models import Intent
from src.infrastructure.repositories import JSONMenuRepository, JSONIntentRepository

def train_classifier():
    """
    Script para entrenar y guardar el clasificador de intenciones
    """
    try:
        # 1. Cargar datos
        print("Cargando datos...")
        base_path = Path(__file__).parent.parent
        menu_repo = JSONMenuRepository(str(base_path / "data" / "menu.json"))
        intent_repo = JSONIntentRepository(str(base_path / "data" / "intents.json"))

        # 2. Inicializar NLPProcessor
        print("Inicializando NLP Processor...")
        nlp_processor = NLPProcessor(
            menu_repo.get_menu(),
            intent_repo.get_intents_dict(),
            str(base_path / "models" / "vectorizer.joblib")
        )

        # 3. Crear y entrenar clasificador
        print("Creando clasificador...")
        classifier = IntentClassifier(nlp_processor)

        print("Entrenando clasificador...")
        intents = intent_repo.get_intents()
        metrics = classifier.train(intents)

        # 4. Guardar modelo entrenado
        print("Guardando modelo...")
        model_dir = base_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        classifier.save_model(str(model_dir / "classifier.joblib"))

        # 5. Imprimir métricas
        print("\nMétricas de entrenamiento:")
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print("\nReporte detallado:")
        for intent, scores in metrics.report.items():
            if isinstance(scores, dict):
                print(f"\nIntent: {intent}")
                print(f"Precision: {scores['precision']:.4f}")
                print(f"Recall: {scores['recall']:.4f}")
                print(f"F1-score: {scores['f1-score']:.4f}")

        print("\nEntrenamiento completado exitosamente!")
        return 0

    except Exception as e:
        print(f"Error durante el entrenamiento: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(train_classifier())