# src/domain/pricing/price_calculator.py
from dataclasses import dataclass
from typing import List, Dict
from src.domain.models import MenuItem, OrderItem

@dataclass
class PriceBreakdown:
    base_price: float
    customization_prices: Dict[str, float]
    total: float

class PriceCalculator:
    def __init__(self):
        # Precios adicionales por tipo de personalización
        self.customization_prices = {
            "leche": {
                "almendra": 0.50,
                "soya": 0.50,
                "avellana": 0.50
            },
            "shots": {
                "extra": 0.75,
                "doble": 1.00
            },
            "syrups": {
                "vainilla": 0.50,
                "caramelo": 0.50,
                "avellana": 0.50
            }
        }

    def calculate_item_price(self, item: OrderItem) -> PriceBreakdown:
        """Calcula el precio detallado de un item del pedido"""
        # Obtener precio base según tamaño
        base_price = item.menu_item.get_base_price(item.size)

        # Calcular precio de personalizaciones
        customization_prices = {}
        for customization in item.customizations:
            category, option = customization.split(":")
            if category in self.customization_prices:
                if option in self.customization_prices[category]:
                    price = self.customization_prices[category][option]
                    customization_prices[customization] = price * item.quantity

        # Calcular total
        total = base_price * item.quantity + sum(customization_prices.values())

        return PriceBreakdown(
            base_price=base_price * item.quantity,
            customization_prices=customization_prices,
            total=total
        )

    def calculate_order_total(self, order_items: List[OrderItem]) -> float:
        """Calcula el total de una orden completa"""
        return sum(self.calculate_item_price(item).total for item in order_items)

    def format_price_options(self, menu_item: MenuItem) -> str:
        """Formatea las opciones de precio para un item del menú"""
        price_text = ["Precios disponibles:"]

        # Precios por tamaño
        for size, price in menu_item.prices.items():
            price_text.append(f"- {size.capitalize()}: ${price:.2f}")

        # Precios de personalizaciones
        if menu_item.customizations:
            price_text.append("\nPersonalizaciones disponibles:")
            for category, options in menu_item.customizations.items():
                if category in self.customization_prices:
                    for option in options:
                        if option in self.customization_prices[category]:
                            extra = self.customization_prices[category][option]
                            price_text.append(f"- {option.capitalize()}: +${extra:.2f}")

        return "\n".join(price_text)