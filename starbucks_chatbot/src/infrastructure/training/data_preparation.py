# src/infrastructure/training/data_preparation.py
from src.domain.training.interfaces import IDataPreparation
from typing import Dict, List
import spacy

class DataPreparation(IDataPreparation):
    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")

    def prepare_menu_data(self, menu_data: Dict) -> List[str]:
        corpus = []

        # Procesar bebidas y alimentos
        for categoria_tipo in ['bebidas', 'alimentos']:
            if categoria_tipo in menu_data:
                for categoria in menu_data[categoria_tipo].values():
                    for item in categoria:
                        # Agregar keywords originales
                        corpus.extend(item["keywords"])

                        # Generar variaciones
                        for keyword in item["keywords"]:
                            corpus.append(keyword.lower())
                            # Agregar nombre del producto también
                            corpus.append(item["nombre"].lower())

        return list(set(corpus))

    def prepare_intent_data(self, intent_data: List) -> List[str]:
        corpus = []

        for intent in intent_data:
            # Agregar ejemplos de entrenamiento
            corpus.extend(intent.get("examples", []))

            # Si hay entidades, agregar sus valores
            entities = intent.get("entities", {})
            for entity_values in entities.values():
                corpus.extend(entity_values)

        return corpus

    def combine_datasets(self, menu_corpus: List[str], intent_corpus: List[str]) -> List[str]:
        combined = menu_corpus + intent_corpus

        # Normalizar textos
        normalized = []
        for text in combined:
            doc = self.nlp(text.lower())
            tokens = [token.text for token in doc
                     if not token.is_stop and not token.is_punct]
            normalized.append(" ".join(tokens))

        return normalized