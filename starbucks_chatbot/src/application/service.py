#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from src.domain.interfaces import IMenuRepository, IIntentRepository
from src.infrastructure.nlp_processor import NLPProcessor, ExtractedEntities
from src.infrastructure.classifier import IntentClassifier
from src.domain.models import Order, OrderItem, MenuItem
from src.application.models import Response

@dataclass
class ConversationState:
    """Estado actual de la conversación"""
    current_intent: Optional[str] = None
    current_order: Optional[Order] = None
    pending_confirmation: bool = False
    last_entities: Optional[ExtractedEntities] = None
    context: Dict = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}

class ChatbotService:
    def __init__(
        self,
        menu_repo: IMenuRepository,
        intent_repo: IIntentRepository,
        nlp_processor: NLPProcessor,
        intent_classifier: IntentClassifier,
        confidence_threshold: float = 0.5
    ):
        self.menu_repo = menu_repo
        self.intent_repo = intent_repo
        self.nlp_processor = nlp_processor
        self.intent_classifier = intent_classifier
        self.confidence_threshold = confidence_threshold
        self.conversation_state = ConversationState()

    def process_message(self, text: str) -> Response:
        """
        Procesa un mensaje del usuario y genera una respuesta

        Args:
            text: Texto del usuario

        Returns:
            Response con la respuesta apropiada
        """
        try:
            # Extraer entidades y predecir intención
            features, entities = self.nlp_processor.process_input(text)
            intent, confidence = self.intent_classifier.predict(text)

            # Actualizar estado
            self.conversation_state.last_entities = entities

            # Verificar confirmación pendiente
            if self.conversation_state.pending_confirmation:
                if self._is_confirmation(text):
                    return self._handle_confirmation(True)
                elif self._is_cancellation(text):
                    return self._handle_confirmation(False)

            # Manejar la intención si la confianza es suficiente
            if confidence >= self.confidence_threshold:
                self.conversation_state.current_intent = intent
                return self.handle_intent(intent, entities)
            else:
                return self._handle_low_confidence(intent, confidence)

        except Exception as e:
            return Response(
                text="Lo siento, tuve un problema procesando tu mensaje. ¿Podrías reformularlo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
            )

    def handle_intent(self, intent: str, entities: ExtractedEntities) -> Response:
        """
        Maneja una intención específica y genera una respuesta apropiada

        Args:
            intent: Intención identificada
            entities: Entidades extraídas

        Returns:
            Response apropiada para la intención
        """
        intent_handlers = {
            "ordenar_bebida": self._handle_order_intent,
            "preguntar_precio": self._handle_price_intent,
            "consultar_menu": self._handle_menu_intent,
            "preguntar_personalizacion": self._handle_customization_intent,
            "confirmar_orden": self._handle_order_confirmation,
            "cancelar_orden": self._handle_order_cancellation
        }

        handler = intent_handlers.get(intent, self._handle_unknown_intent)
        return handler(entities)

    def _handle_order_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de ordenar una bebida"""
        if not self.conversation_state.current_order:
            self.conversation_state.current_order = Order()

        if entities.bebida:
            item = self.menu_repo.search_item(entities.bebida)
            if item:
                order_item = OrderItem(
                    menu_item=item,
                    size=entities.tamaño or "grande",
                    customizations=entities.personalizaciones or []
                )
                self.conversation_state.current_order.add_item(order_item)

                suggested_actions = ["Confirmar orden", "Agregar más", "Ver orden actual"]
                if entities.personalizaciones:
                    text = f"He agregado un {item.name} {order_item.size} con {', '.join(entities.personalizaciones)}. "
                else:
                    text = f"He agregado un {item.name} {order_item.size}. "

                text += "¿Deseas agregar algo más?"

                return Response(
                    text=text,
                    suggested_actions=suggested_actions,
                    order=self.conversation_state.current_order
                )
            else:
                return Response(
                    text="Lo siento, no encontré esa bebida en nuestro menú. ¿Te gustaría ver las opciones disponibles?",
                    suggested_actions=["Ver menú", "Ver bebidas populares"]
                )

        return Response(
            text="¿Qué bebida te gustaría ordenar?",
            suggested_actions=["Ver menú", "Ver bebidas populares"]
        )

    def _handle_price_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de consultar precios"""
        if entities.bebida:
            item = self.menu_repo.search_item(entities.bebida)
            if item:
                prices_text = self._format_prices(item)
                return Response(
                    text=f"Los precios para {item.name} son:\n{prices_text}",
                    suggested_actions=["Ordenar bebida", "Ver otras bebidas", "Consultar otro precio"]
                )

        return Response(
            text="¿De qué producto te gustaría saber el precio?",
            suggested_actions=["Ver menú", "Ver bebidas populares"]
        )

    def _handle_menu_intent(self, entities: Optional[ExtractedEntities] = None) -> Response:
        """Maneja la intención de consultar el menú"""
        menu_text = self._format_menu_summary()
        return Response(
            text=f"Este es nuestro menú:\n{menu_text}\n¿Qué te gustaría ordenar?",
            suggested_actions=["Ver bebidas calientes", "Ver bebidas frías", "Ver precios"]
        )

    def _handle_customization_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de preguntar por personalizaciones"""
        if entities.bebida:
            item = self.menu_repo.search_item(entities.bebida)
            if item:
                customization_text = self._format_customizations(item)
                return Response(
                    text=customization_text,
                    suggested_actions=["Ordenar bebida", "Ver otras opciones"]
                )

        return Response(
            text="Estas son nuestras opciones de personalización generales:\n" +
                 self._format_general_customizations(),
            suggested_actions=["Ver bebidas", "Hacer un pedido"]
        )

    def _handle_order_confirmation(self, entities: Optional[ExtractedEntities] = None) -> Response:
        """Maneja la confirmación de una orden"""
        if self.conversation_state.current_order:
            order_summary = self._format_order_summary(self.conversation_state.current_order)
            self.conversation_state.pending_confirmation = True
            return Response(
                text=f"Tu orden es:\n{order_summary}\n¿Deseas confirmar esta orden?",
                suggested_actions=["Confirmar", "Modificar", "Cancelar"],
                order=self.conversation_state.current_order
            )

        return Response(
            text="No hay una orden activa. ¿Te gustaría ordenar algo?",
            suggested_actions=["Ver menú", "Ordenar bebida"]
        )

    def _handle_order_cancellation(self, entities: Optional[ExtractedEntities] = None) -> Response:
        """Maneja la cancelación de una orden"""
        self.conversation_state = ConversationState()
        return Response(
            text="He cancelado tu orden. ¿Hay algo más en lo que pueda ayudarte?",
            suggested_actions=["Ver menú", "Ordenar otra cosa"]
        )

    def _handle_unknown_intent(self, entities: Optional[ExtractedEntities] = None) -> Response:
        """Maneja intenciones no reconocidas"""
        return Response(
            text="No estoy seguro de lo que quieres hacer. ¿Podrías ser más específico?",
            suggested_actions=["Ver menú", "Hacer pedido", "Ver precios"]
        )

    def _handle_low_confidence(self, intent: str, confidence: float) -> Response:
        """Maneja casos donde la confianza en la predicción es baja"""
        return Response(
            text="No estoy seguro de entender. ¿Podrías reformular tu solicitud?",
            suggested_actions=["Ver menú", "Hacer pedido", "Ver precios"]
        )

    def _handle_confirmation(self, confirmed: bool) -> Response:
        """Maneja la respuesta a una confirmación pendiente"""
        self.conversation_state.pending_confirmation = False

        if confirmed:
            # Aquí iría la lógica de procesar la orden confirmada
            order = self.conversation_state.current_order
            self.conversation_state = ConversationState()
            return Response(
                text="¡Gracias por tu orden! La estamos preparando. ¿Hay algo más en lo que pueda ayudarte?",
                suggested_actions=["Ordenar algo más", "Ver menú"],
                order=order
            )
        else:
            return Response(
                text="He cancelado la confirmación. ¿Qué te gustaría modificar?",
                suggested_actions=["Modificar orden", "Cancelar orden", "Ver menú"]
            )

    # Métodos auxiliares
    def _is_confirmation(self, text: str) -> bool:
        """Verifica si el texto es una confirmación"""
        confirmations = ["si", "sí", "confirmar", "confirmo", "ok", "dale", "listo"]
        return any(conf in text.lower() for conf in confirmations)

    def _is_cancellation(self, text: str) -> bool:
        """Verifica si el texto es una cancelación"""
        cancellations = ["no", "cancelar", "cancela", "mejor no"]
        return any(cancel in text.lower() for cancel in cancellations)

    def _format_prices(self, item: MenuItem) -> str:
        """Formatea los precios de un item para mostrar"""
        return "\n".join([f"- {size}: ${price:.2f}" for size, price in item.prices.items()])

    def _format_menu_summary(self) -> str:
        """Formatea un resumen del menú"""
        menu = self.menu_repo.get_menu()
        summary = []

        for category, items in menu["bebidas"].items():
            category_items = [item["nombre"] for item in items]
            summary.append(f"{category.title()}:\n- " + "\n- ".join(category_items))

        return "\n\n".join(summary)

    def _format_customizations(self, item: MenuItem) -> str:
        """Formatea las opciones de personalización de un item"""
        customization_text = f"Opciones de personalización para {item.name}:\n"

        for category, options in item.customizations.items():
            customization_text += f"\n{category.title()}:\n- " + "\n- ".join(options)

        return customization_text

    def _format_general_customizations(self) -> str:
        """Formatea las opciones generales de personalización"""
        return "- Leches: entera, descremada, almendra, soya\n" + \
               "- Shots de café: simple, doble\n" + \
               "- Jarabes: vainilla, caramelo, avellana"

    def _format_order_summary(self, order: Order) -> str:
        """Formatea un resumen de la orden"""
        summary = []
        for item in order.items:
            item_text = f"- {item.menu_item.name} ({item.size})"
            if item.customizations:
                item_text += f" con {', '.join(item.customizations)}"
            summary.append(item_text)

        summary.append(f"\nTotal: ${order.total:.2f}")
        return "\n".join(summary)