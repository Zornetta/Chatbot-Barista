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
        """Carga los datos del menú desde el archivo JSON"""
        if self._menu_data is None:
            try:
                base_path = Path(__file__).parent
                file_path = (base_path / self.file_path).resolve()
                # Imprimir la ruta que está intentando cargar
                print(f"\nCargando menú desde: {self.file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    self._menu_data = json.load(f)

                print("- Menú cargado exitosamente")
                # Imprimir las categorías disponibles
                print(f"- Categorías disponibles: {list(self._menu_data['bebidas'].keys())}")
            except Exception as e:
                print(f"Error cargando el menú: {str(e)}")
                raise

    def get_menu(self) -> Dict:
        """Obtiene todo el menú"""
        self._load_data()
        return self._menu_data

    def search_item(self, query: str) -> Optional[MenuItem]:
        """Busca un item específico en la sección de bebidas del menú"""
        self._load_data()
        query = query.lower()

        print(f"\nBuscando en el menú de bebidas:")
        print(f"- Query: {query}")

        # Buscar en todas las categorías de bebidas
        for category_name, category in self._menu_data['bebidas'].items():
            print(f"- Buscando en categoría: {category_name}")
            for item in category:
                print(f"  - Comparando con item: {item['id']}")

                # Comparar directamente con el ID
                if query == item['id']:
                    print(f"  - ¡Coincidencia encontrada por ID!")
                    return self._create_beverage_menu_item(item)

                # Buscar en keywords
                if any(keyword.lower() == query for keyword in item['keywords']):
                    print(f"  - ¡Coincidencia encontrada por keyword!")
                    return self._create_beverage_menu_item(item)

        print(f"- No se encontraron coincidencias para: {query}")
        return None

    def search_food_item(self, query: str) -> Optional[MenuItem]:
        """Busca un item específico en la sección de alimentos del menú"""
        self._load_data()
        query = query.lower()

        print(f"\nBuscando alimento en el menú:")
        print(f"- Query: {query}")

        # Buscar en todas las categorías de alimentos
        if 'alimentos' in self._menu_data:
            for category_name, category in self._menu_data['alimentos'].items():
                print(f"- Buscando en categoría: {category_name}")
                for item in category:
                    print(f"  - Comparando con item: {item['id']}")

                    # Comparar directamente con el ID
                    if query == item['id']:
                        print(f"  - ¡Coincidencia encontrada por ID!")
                        return self._create_food_menu_item(item)

                    # Buscar en keywords
                    if any(keyword.lower() == query for keyword in item['keywords']):
                        print(f"  - ¡Coincidencia encontrada por keyword!")
                        return self._create_food_menu_item(item)

        print(f"- No se encontraron coincidencias para: {query}")
        return None

    def _create_beverage_menu_item(self, item_data: Dict) -> MenuItem:
        """Crea un MenuItem para bebidas con precios y calorías por tamaño"""
        return MenuItem(
            id=item_data['id'],
            name=item_data['nombre'],
            category=item_data['categoria'],
            sizes=item_data['tamaños'],
            prices=item_data['precios'],
            calories=item_data.get('calorias', {
                'tall': 0,
                'grande': 0,
                'venti': 0
            }),  # Manejo de calorías por tamaño para bebidas
            customizations=item_data.get('personalizaciones', {}),
            keywords=item_data['keywords']
        )

    def _create_food_menu_item(self, item_data: Dict) -> MenuItem:
        """Crea un MenuItem para alimentos con precio único y calorías por porción"""
        # Para alimentos, convertir precio único a diccionario por tamaño
        size = item_data.get('tamaños', ['individual'])[0]
        prices = {size: item_data.get('precio', 0)} if 'precio' in item_data else item_data.get('precios', {})

        return MenuItem(
            id=item_data['id'],
            name=item_data['nombre'],
            category=item_data['categoria'],
            sizes=item_data.get('tamaños', ['individual']),
            prices=prices,
            calories=item_data.get('calorias', {size: 0}),  # Manejo de calorías para porción individual
            customizations=item_data.get('personalizaciones', {}),
            keywords=item_data['keywords']
        )

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