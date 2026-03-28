"""
app/main.py — Punto de entrada de la API de Dominó.

Arrancar con:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Documentación interactiva disponible en:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import game, benchmark, metrics

app = FastAPI(
    title="Dominó AI — API",
    description=(
        "API para simulación y análisis de un juego de dominó con múltiples "
        "algoritmos de búsqueda: Minimax + poda α-β, A*, distancias Manhattan "
        "y Euclidiana, y un agente Híbrido. Incluye streaming SSE para "
        "visualización en tiempo real de métricas de eficiencia."
    ),
    version="1.0.0",
)

# ── CORS — permitir que el frontend acceda a la API ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En producción: reemplazar con el origen del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
app.include_router(game.router,      prefix="/api/game",      tags=["Juego"])
app.include_router(benchmark.router, prefix="/api/benchmark", tags=["Benchmark"])
app.include_router(metrics.router,   prefix="/api/metrics",   tags=["Métricas"])


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Dominó AI API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "new_game":        "POST /api/game/new",
            "step_turn":       "POST /api/game/{id}/step",
            "stream_game":     "GET  /api/game/{id}/stream  (SSE)",
            "get_state":       "GET  /api/game/{id}",
            "turn_history":    "GET  /api/game/{id}/history",
            "realtime_charts": "GET  /api/metrics/game/{id}/realtime",
            "endgame_charts":  "GET  /api/metrics/game/{id}/summary",
            "run_benchmark":   "POST /api/benchmark/run     (SSE)",
            "strategies":      "GET  /api/metrics/strategies",
        },
    }
