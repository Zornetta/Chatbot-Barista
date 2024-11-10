#!/usr/bin/env python3

# src/domain/models.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class MenuItem:
    id: str
    name: str
    category: str
    sizes: List[str]
    prices: Dict[str, float]
    customizations: Dict[str, List[str]]
    keywords: List[str]

    def get_base_price(self, size: str) -> float:
        """Obtiene el precio base para un tamaño específico"""
        return self.prices.get(size, 0.0)

    def is_customization_valid(self, customization_type: str, value: str) -> bool:
        """Verifica si una personalización es válida para este item"""
        valid_options = self.customizations.get(customization_type, [])
        return value in valid_options

@dataclass
class Intent:
    name: str
    examples: List[str]
    entities: Dict[str, List[str]]

    def matches_example(self, text: str) -> bool:
        """Verifica si el texto coincide con algún ejemplo"""
        return any(example.lower() in text.lower() for example in self.examples)

@dataclass
class OrderItem:
    menu_item: MenuItem
    size: str
    customizations: List[str]
    quantity: int = 1

    def calculate_price(self) -> float:
        """Calcula el precio total del item incluyendo personalizaciones"""
        base_price = self.menu_item.get_base_price(self.size)
        # Aquí se podría agregar lógica para calcular costos adicionales por personalizaciones
        return base_price * self.quantity

@dataclass
class Order:
    items: List[OrderItem] = field(default_factory=list)
    total: float = 0.0

    def add_item(self, item: OrderItem) -> None:
        """Agrega un item al pedido"""
        self.items.append(item)
        self.calculate_total()

    def remove_item(self, item: OrderItem) -> None:
        """Elimina un item del pedido"""
        self.items.remove(item)
        self.calculate_total()

    def calculate_total(self) -> float:
        """Calcula el total del pedido"""
        self.total = sum(item.calculate_price() for item in self.items)
        return self.total