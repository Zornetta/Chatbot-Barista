#!/usr/bin/env python3

# src/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from src.domain.models import MenuItem, Intent

class IMenuRepository(ABC):
    @abstractmethod
    def get_menu(self) -> Dict:
        """Obtiene todo el menú"""
        pass

    @abstractmethod
    def search_item(self, query: str) -> Optional['MenuItem']:
        """Busca un item específico en el menú"""
        pass

class IIntentRepository(ABC):
    @abstractmethod
    def get_intents(self) -> List['Intent']:
        """Obtiene todas las intenciones"""
        pass

    @abstractmethod
    def get_intents_dict(self) -> Dict:
        """Obtiene todas las intenciones comom un dict"""
        pass

    @abstractmethod
    def get_intent_by_name(self, name: str) -> Optional['Intent']:
        """Busca una intención específica por nombre"""
        pass