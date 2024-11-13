#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
from src.domain.interfaces import IMenuRepository, IIntentRepository
from src.infrastructure.nlp_processor import NLPProcessor, ExtractedEntities
from src.infrastructure.classifier import IntentClassifier
from src.domain.models import Order, OrderItem, MenuItem
from src.application.models import Response
from src.domain.pricing.price_calculator import PriceCalculator, PriceBreakdown

class InteractionMode(Enum):
    PURCHASE = "purchase"  # Modo compra
    QUERY = "query"       # Modo consulta

@dataclass
class ConversationState:
    """Estado actual de la conversación"""
    current_intent: Optional[str] = None
    predicted_intent: Optional[str] = None
    current_order: Optional[Order] = None
    pending_confirmation: bool = False
    pending_intent_confirmation: bool = False
    last_entities: Optional[ExtractedEntities] = None
    last_input: Optional[str] = None
    mode: InteractionMode = InteractionMode.QUERY  # Empezamos en modo consulta por defecto
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
        self.price_calculator = PriceCalculator()

        # Mapeo de intenciones a descripciones amigables
        self.intent_descriptions = {
            "ordenar_bebida": "realizar un pedido de bebida",
            "ordenar_alimento": "realizar un pedido de comida",
            "preguntar_precio": "consultar precios",
            "consultar_menu": "ver el menú"
        }

        # Intenciones que no necesitan confirmación
        self.direct_intents = {
            "consultar_menu",
            "confirmar_orden",
            #"cancelar_orden"
        }

    def process_message(self, text: str) -> Response:
        """Procesa un mensaje del usuario y genera una respuesta"""
        try:
            print(f"\nProcesando mensaje: '{text}'")
            print(f"Modo actual: {self.conversation_state.mode}")

            # Procesar el input
            features, entities = self.nlp_processor.process_input(text)
            intent, confidence = self.intent_classifier.predict(text)

            print(f"Intent detectado: {intent}")
            print(f"Entidades encontradas: {entities}")

            # Si es una intención directa, procesarla inmediatamente
            if intent in self.direct_intents:
                print(f"Procesando intención directa: {intent}")
                return self.handle_intent(intent, entities)

            # Actualizar estado
            self.conversation_state.last_entities = entities
            self.conversation_state.last_input = text
            self.conversation_state.predicted_intent = intent

            # Manejar las intenciones según el modo actual
            if intent in ["ordenar_bebida", "ordenar_alimento"]:
                self.conversation_state.mode = InteractionMode.PURCHASE
                return self._handle_purchase_intent(intent, entities)
            elif intent == "consultar_menu":
                self.conversation_state.mode = InteractionMode.QUERY
                return self._handle_menu_intent(entities)
            elif intent == "preguntar_precio":
                self.conversation_state.mode = InteractionMode.QUERY
                return self._handle_price_query(text, entities)
            else:
                return self._handle_unknown_intent(entities)

        except Exception as e:
            print(f"Error en process_message: {str(e)}")
            return Response(
                text="Lo siento, tuve un problema procesando tu mensaje. ¿Podrías reformularlo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
            )

    def handle_intent(self, intent: str, entities: ExtractedEntities) -> Response:
        """Maneja una intención específica y genera una respuesta apropiada"""
        intent_handlers = {
            "ordenar_bebida": self._handle_purchase_intent,
            "ordenar_alimento": self._handle_purchase_intent,
            "preguntar_precio": self._handle_price_intent,
            "consultar_menu": self._handle_menu_intent,
            "confirmar_orden": self._handle_order_confirmation,  # Handler específico para confirmación
            "cancelar_orden": self._handle_order_cancellation
        }

        handler = intent_handlers.get(intent, self._handle_unknown_intent)
        return handler(intent, entities)

    def _handle_purchase_intent(self, intent: str, entities: ExtractedEntities) -> Response:
        """Maneja las intenciones de compra (ordenar_bebida, ordenar_alimento)"""
        if intent == "ordenar_bebida":
            return self._handle_order_intent(entities)
        else:  # ordenar_alimento
            return self._handle_food_order_intent(entities)

    def _handle_order_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de ordenar una bebida"""
        try:
            if entities and entities.bebida:
                item = self.menu_repo.search_item(entities.bebida)
                if item:
                    # Mostrar primero los precios disponibles si no se especificó tamaño
                    if not entities.tamaño:
                        price_options = self.price_calculator.format_price_options(item)
                        return Response(
                            text=f"Encontré {item.name}. {price_options}\n\n¿Qué tamaño prefieres?",
                            suggested_actions=item.sizes,
                            order=self.conversation_state.current_order
                        )

                    # Si hay tamaño, crear el item y mostrar precio
                    order_item = OrderItem(
                        menu_item=item,
                        size=entities.tamaño,
                        customizations=entities.personalizaciones or []
                    )

                    # Calcular desglose de precios
                    price_breakdown = self.price_calculator.calculate_item_price(order_item)

                    # Agregar a la orden si existe
                    if not self.conversation_state.current_order:
                        self.conversation_state.current_order = Order()
                    self.conversation_state.current_order.add_item(order_item)

                    # Preparar mensaje con desglose de precios
                    price_text = f"\nPrecio base: ${price_breakdown.base_price:.2f}"
                    if price_breakdown.customization_prices:
                        price_text += "\nPersonalizaciones:"
                        for custom, price in price_breakdown.customization_prices.items():
                            price_text += f"\n- {custom}: +${price:.2f}"
                    price_text += f"\nTotal: ${price_breakdown.total:.2f}"

                    text = f"He agregado un {item.name} {order_item.size}"
                    if entities.personalizaciones:
                        text += f" con {', '.join(entities.personalizaciones)}"
                    text += f".{price_text}\n\n¿Deseas agregar algo más?"

                    return Response(
                        text=text,
                        suggested_actions=["Ver personalizaciones", "Confirmar orden", "Agregar más"],
                        order=self.conversation_state.current_order
                    )

            return Response(
                text="¿Qué bebida te gustaría ordenar? Puedo mostrarte nuestro menú con precios.",
                suggested_actions=["Ver menú", "Ver bebidas populares"]
            )

        except Exception as e:
            print(f"Error en _handle_order_intent: {str(e)}")
            return Response(
                text="Lo siento, ocurrió un error al procesar tu orden. ¿Podrías intentarlo de nuevo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
            )

        except Exception as e:
            print(f"\nError en _handle_order_intent: {str(e)}")
            return Response(
                text="Lo siento, ocurrió un error al procesar tu orden. ¿Podrías intentarlo de nuevo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
            )

    def _handle_food_order_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de ordenar comida"""
        try:
            # Debug: Imprimir estado al manejar orden
            print(f"\nManejando orden de comida:")
            print(f"- Entidades recibidas: {entities}")

            if not self.conversation_state.current_order:
                self.conversation_state.current_order = Order()

            if entities and entities.alimento:  # Nueva propiedad para alimentos
                print(f"- Buscando alimento: {entities.alimento}")
                item = self.menu_repo.search_food_item(entities.alimento)  # Nuevo método
                print(f"- Item encontrado: {item}")

                if item:
                    order_item = OrderItem(
                        menu_item=item,
                        size="individual",  # La mayoría de alimentos son tamaño individual
                        customizations=entities.personalizaciones or []
                    )
                    self.conversation_state.current_order.add_item(order_item)

                    print(f"- Item agregado a la orden: {order_item.menu_item.name}")

                    suggested_actions = ["Confirmar orden", "Agregar más", "Ver orden actual"]
                    if entities.personalizaciones:
                        text = f"He agregado {item.name} con {', '.join(entities.personalizaciones)}. "
                    else:
                        text = f"He agregado {item.name}. "

                    text += "¿Deseas agregar algo más?"

                    return Response(
                        text=text,
                        suggested_actions=suggested_actions,
                        order=self.conversation_state.current_order
                    )
                else:
                    print(f"- No se encontró el alimento en el menú")
                    return Response(
                        text="Lo siento, no encontré ese alimento en nuestro menú. ¿Te gustaría ver las opciones disponibles?",
                        suggested_actions=["Ver menú", "Ver comidas disponibles"]
                    )

            print(f"- No se proporcionó alimento en las entidades")
            return Response(
                text="¿Qué te gustaría ordenar de comer?",
                suggested_actions=["Ver menú de comidas", "Ver opciones populares"]
            )

        except Exception as e:
            print(f"\nError en _handle_food_order_intent: {str(e)}")
            return Response(
                text="Lo siento, ocurrió un error al procesar tu orden. ¿Podrías intentarlo de nuevo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
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
            text=f"Este es nuestro menú:\n{menu_text}\n\n¿Qué te gustaría hacer?",
            suggested_actions=[
                "Ordenar algo",
                "Consultar precios",
                "Ver bebidas",
                "Ver alimentos"
            ]
        )

    def _handle_customization_intent(self, entities: ExtractedEntities) -> Response:
        """Maneja la intención de preguntar por personalizaciones"""
        if entities.bebida:
            item = self.menu_repo.search_item(entities.bebida)
            if item:
                customization_text = self.price_calculator.format_price_options(item)
                return Response(
                    text=f"Estas son las opciones para {item.name}:\n{customization_text}",
                    suggested_actions=["Ordenar bebida", "Ver otras opciones"]
                )

        # Mostrar opciones generales con precios
        customization_text = "Opciones de personalización disponibles:\n"
        for category, options in self.price_calculator.customization_prices.items():
            customization_text += f"\n{category.capitalize()}:"
            for option, price in options.items():
                customization_text += f"\n- {option.capitalize()}: +${price:.2f}"

        return Response(
            text=customization_text,
            suggested_actions=["Ver bebidas", "Hacer un pedido"]
        )

    def _handle_order_confirmation(self, intent: str, entities: ExtractedEntities) -> Response:
        """Maneja la confirmación directa de una orden"""
        if self.conversation_state.current_order:
            order = self.conversation_state.current_order
            # Limpiar el estado de la conversación
            self.conversation_state = ConversationState()

            return Response(
                text=f"¡Gracias por tu orden! Tu pedido está siendo preparado:\n" +
                     self._format_order_summary(order) +
                     "\n\n¿Hay algo más en lo que pueda ayudarte?",
                suggested_actions=["Ordenar algo más", "Ver menú"],
                order=order
            )

        return Response(
            text="No hay una orden activa para confirmar. ¿Te gustaría ordenar algo?",
            suggested_actions=["Ver menú", "Hacer un pedido"]
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

    def _handle_intent_confirmation(self, text: str) -> Response:
        """
        Maneja la confirmación de la intención detectada
        """
        try:
            # Debug: Imprimir estado antes de procesar
            print(f"\nManejando confirmación de intención:")
            print(f"- Texto de confirmación: {text}")
            print(f"- Intent pendiente: {self.conversation_state.predicted_intent}")
            print(f"- Entidades guardadas: {self.conversation_state.last_entities}")

            self.conversation_state.pending_intent_confirmation = False

            if self._is_confirmation(text):
                # Proceder con la intención confirmada
                intent = self.conversation_state.predicted_intent
                entities = self.conversation_state.last_entities

                print(f"\nConfirmación aceptada:")
                print(f"- Intent a procesar: {intent}")
                print(f"- Entidades a usar: {entities}")

                self.conversation_state.current_intent = intent
                return self.handle_intent(intent, entities)
            else:
                # Solicitar nuevo input
                return Response(
                    text="Entiendo. ¿Podrías reformular tu solicitud?",
                    suggested_actions=["Ver menú", "Ver opciones disponibles"]
                )

        except Exception as e:
            print(f"\nError en _handle_intent_confirmation: {str(e)}")
            return Response(
                text="Lo siento, ocurrió un error al procesar tu confirmación. ¿Podrías intentarlo de nuevo?",
                suggested_actions=["Ver menú", "Empezar de nuevo"]
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
        if isinstance(item.prices, dict):
            return "\n".join([f"- {size}: ${price:.2f}" for size, price in item.prices.items()])
        else:
            return f"${item.prices:.2f}"

    def _format_menu_summary(self) -> str:
        """Formatea un resumen del menú incluyendo bebidas y alimentos"""
        menu = self.menu_repo.get_menu()
        summary = []

        # Agregar bebidas
        for category, items in menu["bebidas"].items():
            category_items = [item["nombre"] for item in items]
            summary.append(f"{category.title()}:\n- " + "\n- ".join(category_items))

        # Agregar alimentos
        if "alimentos" in menu:
            summary.append("\nAlimentos:")
            for category, items in menu["alimentos"].items():
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
        """Formatea un resumen de la orden con desglose de precios"""
        summary = []
        for item in order.items:
            breakdown = self.price_calculator.calculate_item_price(item)

            item_text = f"- {item.menu_item.name} ({item.size})"
            if item.customizations:
                item_text += f" con {', '.join(item.customizations)}"

            # Agregar desglose de precios
            item_text += f"\n  Base: ${breakdown.base_price:.2f}"
            for custom, price in breakdown.customization_prices.items():
                item_text += f"\n  {custom}: +${price:.2f}"
            item_text += f"\n  Subtotal: ${breakdown.total:.2f}"

            summary.append(item_text)

        # Agregar total general
        total = self.price_calculator.calculate_order_total(order.items)
        summary.append(f"\nTotal Final: ${total:.2f}")

        return "\n".join(summary)

    def _handle_price_query(self, text: str, entities: ExtractedEntities) -> Response:
        """Maneja las consultas de precio"""
        print("\nProcesando consulta de precio")

        # Intentar identificar producto usando los procesadores de ordenar_bebida/alimento
        product = None

        # Primero intentar con bebidas
        drink_features, drink_entities = self.nlp_processor.process_input(text)
        if drink_entities.bebida:
            print(f"Bebida identificada: {drink_entities.bebida}")
            product = self.menu_repo.search_item(drink_entities.bebida)

        # Si no encontró bebida, intentar con alimentos
        if not product and drink_entities.alimento:
            print(f"Alimento identificado: {drink_entities.alimento}")
            product = self.menu_repo.search_food_item(drink_entities.alimento)

        if product:
            # Mostrar precio y ofrecer compra
            prices_text = self._format_prices(product)
            return Response(
                text=f"Los precios para {product.name} son:\n{prices_text}\n\n¿Te gustaría ordenarlo?",
                suggested_actions=[
                    f"Ordenar {product.name}",
                    "Consultar otro precio",
                    "Ver menú completo"
                ]
            )
        else:
            # No se identificó producto, pedir especificación
            return Response(
                text="¿De qué producto te gustaría saber el precio?",
                suggested_actions=[
                    "Ver bebidas disponibles",
                    "Ver alimentos disponibles",
                    "Ver menú completo"
                ]
            )
