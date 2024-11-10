#!/usr/bin/env python3

# src/application/models.py
from dataclasses import dataclass
from typing import List, Optional
from src.domain.models import Order

@dataclass
class Response:
    text: str
    suggested_actions: List[str]
    order: Optional[Order] = None