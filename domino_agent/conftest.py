"""
conftest.py — Fixtures compartidas entre todos los tests.
"""
import sys
import os
import pytest
from httpx import AsyncClient, ASGITransport

# Asegurar que 'domino_agent/' esté en el path para imports absolutos
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Cliente HTTP asíncrono contra la app FastAPI en memoria."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def game_state():
    """Estado de partida nueva para tests unitarios."""
    from app.core.game_state import GameState
    return GameState.new_game()
