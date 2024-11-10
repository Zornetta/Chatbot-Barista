#!/usr/bin/env python3

# src/infrastructure/repositories.py
import json
from typing import Dict, List, Optional
from src.domain.interfaces import IMenuRepository, IIntentRepository
from src.domain.models import MenuItem, Intent
from pathlib import Path

class JSONMenuRepository(IMenuRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._menu_data = None

    def _load_data(self):
        base_path = Path(__file__).parent
        file_path = (base_path / self.file_path).resolve()

        if self._menu_data is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._menu_data = json.load(f)

    def get_menu(self) -> Dict:
        self._load_data()
        return self._menu_data

    def search_item(self, query: str) -> Optional[MenuItem]:
        self._load_data()
        query = query.lower()

        # Buscar en todas las categorías
        for category in self._menu_data['bebidas'].values():
            for item in category:
                if query in [k.lower() for k in item['keywords']]:
                    return MenuItem(
                        id=item['id'],
                        name=item['nombre'],
                        category=item['categoria'],
                        sizes=item['tamaños'],
                        prices=item['precios'],
                        customizations=item['personalizaciones'],
                        keywords=item['keywords']
                    )
        return None

class JSONIntentRepository(IIntentRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._intents_data = None

    def _load_data(self):
        base_path = Path(__file__).parent
        file_path = (base_path / self.file_path).resolve()

        if self._intents_data is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._intents_data = json.load(f)

    def get_intents(self) -> List[Intent]:
        self._load_data()
        return [Intent(
            name=intent['intent'],
            examples=intent['examples'],
            entities=intent.get('entities', {})
        ) for intent in self._intents_data]

    def get_intents_dict(self) -> Dict:
        self._load_data()
        return self._intents_data

    def get_intent_by_name(self, name: str) -> Optional[Intent]:
        self._load_data()
        for intent in self._intents_data:
            if intent['intent'] == name:
                return Intent(
                    name=intent['intent'],
                    examples=intent['examples'],
                    entities=intent.get('entities', {})
                )
        return None