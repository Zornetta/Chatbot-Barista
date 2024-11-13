#!/usr/bin/env python3

import spacy
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import joblib

@dataclass
class ExtractedEntities:
    """Clase para almacenar las entidades extraídas del texto"""
    bebida: Optional[str] = None
    tamaño: Optional[str] = None
    personalizaciones: List[str] = None

    def __post_init__(self):
        if self.personalizaciones is None:
            self.personalizaciones = []

class NLPProcessor:
    def __init__(self, menu_data: dict, intents_data: list, model_path: str = None):
        """
        Inicializa el procesador NLP

        Args:
            menu_data: Diccionario con los datos del menú
            intents_data: Lista de intenciones y ejemplos
            model_path: Ruta opcional al modelo vectorizador
        """
        # Cargar modelo de spaCy (español)
        self.nlp = spacy.load("es_core_news_sm")

        # Almacenar datos de referencia
        self.menu_data = menu_data
        self.intents_data = intents_data

        # Cargar vectorizador entrenado
        self.vectorizer = self._load_vectorizer(model_path)

        # Compilar keywords y entidades del menú
        self.bebidas_keywords = self._compile_keywords()
        self.tamaños = ["tall", "grande", "venti"]

    def _load_vectorizer(self, model_path: str = None) -> TfidfVectorizer:
        """
        Carga el vectorizador entrenado desde el archivo

        Args:
            model_path: Ruta opcional al modelo. Si no se proporciona,
                       usa la ruta por defecto

        Returns:
            Vectorizador entrenado

        Raises:
            RuntimeError: Si no se puede cargar el vectorizador
        """
        try:
            if model_path is None:
                base_path = Path(__file__).parent.parent.parent
                model_path = base_path / "models" / "vectorizer.joblib"

            return joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Error al cargar el vectorizador: {str(e)}")

    def _compile_keywords(self) -> Dict[str, List[str]]:
        """Compila todos los keywords de bebidas del menú"""
        keywords = {}
        for categoria in self.menu_data["bebidas"].values():
            for bebida in categoria:
                keywords[bebida["id"]] = bebida["keywords"]
        return keywords

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesa el texto de entrada

        Args:
            text: Texto a procesar

        Returns:
            Texto procesado
        """
        # Procesar con spaCy
        doc = self.nlp(text.lower())

        # Remover stopwords y puntuación
        tokens = [token.text for token in doc
                 if not token.is_stop and not token.is_punct]

        return " ".join(tokens)

    def extract_entities(self, text: str) -> ExtractedEntities:
        """
        Extrae entidades del texto (bebida, tamaño, personalizaciones)

        Args:
            text: Texto de entrada

        Returns:
            ExtractedEntities con las entidades encontradas
        """
        doc = self.nlp(text.lower())
        entities = ExtractedEntities()

        # Buscar bebida
        for bebida_id, keywords in self.bebidas_keywords.items():
            if any(keyword in text.lower() for keyword in keywords):
                entities.bebida = bebida_id
                break

        # Buscar tamaño
        for tamaño in self.tamaños:
            if tamaño in text.lower():
                entities.tamaño = tamaño
                break

        # Buscar personalizaciones
        personalizaciones = []
        for categoria in self.menu_data["bebidas"].values():
            for bebida in categoria:
                for tipo_pers, opciones in bebida["personalizaciones"].items():
                    for opcion in opciones:
                        if opcion in text.lower():
                            personalizaciones.append(f"{tipo_pers}:{opcion}")

        entities.personalizaciones = list(set(personalizaciones))

        return entities

    def prepare_for_classification(self, text: str) -> np.ndarray:
        """
        Prepara el texto para la clasificación usando el vectorizador entrenado

        Args:
            text: Texto preprocesado

        Returns:
            Vector de características
        """
        # Usar el vectorizador entrenado para transformar el texto
        try:
            return self.vectorizer.transform([text])
        except Exception as e:
            raise RuntimeError(f"Error al vectorizar el texto: {str(e)}")

    def process_input(self, text: str) -> Tuple[np.ndarray, ExtractedEntities]:
        """
        Procesa el texto de entrada completo

        Args:
            text: Texto del usuario

        Returns:
            Tupla con (vector para clasificación, entidades extraídas)
        """
        processed_text = self.preprocess_text(text)
        entities = self.extract_entities(text)
        features = self.prepare_for_classification(processed_text)

        return features, entities

    def get_vectorizer(self) -> TfidfVectorizer:
        """
        Obtiene el vectorizador actual

        Returns:
            Vectorizador actual
        """
        return self.vectorizer