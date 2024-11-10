#!/usr/bin/env python3

# src/application/service.py
from src.domain.interfaces import IMenuRepository, IIntentRepository
from src.infrastructure.nlp_processor import NLPProcessor
from src.infrastructure.classifier import IntentClassifier
from src.application.models import Response
from src.domain.models import Order, OrderItem
from typing import Optional, List

class ChatbotService:
    def __init__(
        self,
        menu_repo: IMenuRepository,
        intent_repo: IIntentRepository,
        nlp_processor: NLPProcessor,
        intent_classifier: IntentClassifier
    ):
        self.menu_repo = menu_repo
        self.intent_repo = intent_repo
        self.nlp_processor = nlp_processor
        self.intent_classifier = intent_classifier
        self.current_order: Optional[Order] = None

    def process_message(self, text: str) -> Response:
        """Procesa un mensaje del usuario y genera una respuesta"""
        # Procesar texto
        features, entities = self.nlp_processor.process_input(text)

        # Predecir intención
        intent = self.intent_classifier.predict(text)

        # Manejar la intención
        return self.handle_intent(intent, entities)

    def handle_intent(self, intent: str, entities) -> Response:
        """Maneja una intención específica y genera una respuesta apropiada"""
        if intent == "ordenar_bebida":
            return self._handle_order_intent(entities)
        elif intent == "preguntar_precio":
            return self._handle_price_intent(entities)
        elif intent == "consultar_menu":
            return self._handle_menu_intent()
        # ... más handlers para otras intenciones

        return Response(
            text="Lo siento, no entendí lo que quieres hacer.",
            suggested_actions=["Ver menú", "Hacer pedido", "Consultar precios"]
        )

    def _handle_order_intent(self, entities) -> Response:
        if entities.bebida:
            item = self.menu_repo.search_item(entities.bebida)
            if item:
                if not self.current_order:
                    self.current_order = Order()

                order_item = OrderItem(
                    menu_item=item,
                    size=entities.tamaño or "grande",  # tamaño por defecto
                    customizations=entities.personalizaciones
                )
                self.current_order.add_item(order_item)

                return Response(
                    text=f"He agregado {item.name} a tu orden. ¿Deseas algo más?",
                    suggested_actions=["Ver orden", "Agregar más", "Finalizar pedido"],
                    order=self.current_order
                )

        return Response(
            text="¿Qué bebida te gustaría ordenar?",
            suggested_actions=["Ver menú", "Ver bebidas populares"]
        )

    def create_order(self, items: List[OrderItem]) -> Order:
        """Crea una nueva orden con los items especificados"""
        order = Order()
        for item in items:
            order.add_item(item)
        return order