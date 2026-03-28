# Dominó AI — Backend API

API REST + SSE para la simulación y análisis comparativo de agentes inteligentes en el juego de dominó (doble 6). Diseñada para ser consumida por un frontend que visualice métricas de rendimiento y costo computacional en tiempo real.

---

## Algoritmos implementados

| Estrategia | Descripción |
|---|---|
| `random` | Baseline: jugada aleatoria entre las válidas |
| `manhattan` | Minimax + poda α-β con heurística de distancia Manhattan |
| `euclidean` | Minimax + poda α-β con heurística de distancia Euclidiana |
| `astar` | Búsqueda A* pura (heurística combinada Manhattan + Euclidiana) |
| `hybrid` | A* para filtrar candidatas → Minimax + poda α-β sobre el top-K |

Todos los agentes registran por turno: `time_ms`, `nodes_expanded`, `eval_calls`, `max_depth`.

---

## Estructura del proyecto

```
domino_agent/
├── app/
│   ├── main.py                    # Entrada FastAPI
│   ├── core/
│   │   ├── game_state.py          # Lógica del juego (Tile, GameState)
│   │   ├── evaluator.py           # Heurísticas: Manhattan, Euclidiana, evaluate()
│   │   └── profiler.py            # CostProfiler — métricas por turno
│   ├── strategies/
│   │   ├── base.py                # Clase abstracta AgentStrategy
│   │   ├── random_strategy.py
│   │   ├── manhattan_strategy.py
│   │   ├── euclidean_strategy.py
│   │   ├── astar_strategy.py
│   │   ├── hybrid_strategy.py
│   │   └── __init__.py            # Registro STRATEGIES{}
│   └── api/
│       ├── schemas.py             # Modelos Pydantic (request/response)
│       ├── game_manager.py        # Sesiones en memoria
│       └── routes/
│           ├── game.py            # Endpoints de partida
│           ├── benchmark.py       # Torneo entre estrategias
│           └── metrics.py         # Datos para gráficas
├── tests/
│   ├── test_game_state.py         # 18 tests del núcleo del juego
│   ├── test_evaluator.py          # 13 tests de heurísticas
│   ├── test_profiler.py           # 11 tests del profiler
│   ├── test_strategies.py         # 29 tests de las 5 estrategias
│   └── test_api.py                # 15 tests de integración HTTP
├── conftest.py                    # Fixtures pytest compartidas
├── pytest.ini                     # Configuración de pytest
└── requirements.txt
```

---

## Requisitos previos

- Python 3.12 o superior
- `py` launcher (incluido en instalaciones estándar de Python en Windows)

---

## Despliegue local

### 1. Clonar el repositorio

```powershell
git clone <url-del-repositorio>
cd Domin-MINIMAX_PodaA-B_Aest\domino_agent
```

### 2. Crear el entorno virtual

```powershell
py -3.12 -m venv .venv
```

### 3. Activar el entorno virtual

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

### 4. Instalar dependencias

```powershell
pip install -r requirements.txt
```

### 5. Levantar la API

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API queda disponible en:

| URL | Descripción |
|---|---|
| `http://localhost:8000/docs` | Swagger UI interactivo |
| `http://localhost:8000/redoc` | Documentación ReDoc |
| `http://localhost:8000/` | Info de endpoints disponibles |

---

## Referencia de endpoints

### Juego (`/api/game`)

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/game/new` | Crear nueva sesión de partida |
| `GET` | `/api/game/` | Listar todas las sesiones activas |
| `GET` | `/api/game/{id}` | Estado actual de la partida |
| `POST` | `/api/game/{id}/step` | Ejecutar exactamente un turno |
| `GET` | `/api/game/{id}/stream` | **SSE**: auto-jugar y emitir eventos por turno |
| `GET` | `/api/game/{id}/history` | Historial completo de turnos jugados |
| `DELETE` | `/api/game/{id}` | Eliminar sesión |

**Ejemplo — crear partida:**
```bash
curl -X POST http://localhost:8000/api/game/new \
  -H "Content-Type: application/json" \
  -d '{"strategy_a": "hybrid", "strategy_b": "random"}'
```

**Ejemplo — streaming SSE desde JavaScript:**
```js
const es = new EventSource(`http://localhost:8000/api/game/{id}/stream?delay_ms=300`);
es.onmessage = (e) => {
  const event = JSON.parse(e.data);
  if (event.type === "turn") {
    // actualizar gráficas en tiempo real con event.metrics
  }
  if (event.type === "game_over") {
    es.close();
    // mostrar resumen final
  }
};
```

### Benchmark (`/api/benchmark`)

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/benchmark/matchups` | Matchups estándar disponibles |
| `POST` | `/api/benchmark/run` | **SSE**: torneo completo con progreso en tiempo real |

**Ejemplo — lanzar torneo de 20 partidas por matchup:**
```bash
curl -X POST http://localhost:8000/api/benchmark/run \
  -H "Content-Type: application/json" \
  -d '{"n_games": 20}'
```

### Métricas (`/api/metrics`)

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/api/metrics/strategies` | Descripción de todas las estrategias |
| `GET` | `/api/metrics/game/{id}` | Métricas completas turno a turno |
| `GET` | `/api/metrics/game/{id}/realtime` | Datos listos para gráficas en tiempo real |
| `GET` | `/api/metrics/game/{id}/summary` | Datos listos para gráficas post-partida |

---

## Gráficas: ¿cuándo mostrarlas?

### En tiempo real (durante la partida) — vía SSE o polling a `/realtime`

Estas métricas tienen valor instantáneo turno a turno y son ideales para visualización en vivo:

| Gráfica | Tipo sugerido | Métrica |
|---|---|---|
| Tiempo de decisión por turno | Línea | `time_ms` |
| Nodos expandidos por turno | Línea | `nodes_expanded` |
| Llamadas a heurística | Barras | `eval_calls` |
| Profundidad de búsqueda | Escalón (`step`) | `max_depth` |
| Fichas en mano | Línea doble | `hand_size_a / b` |
| Longitud del tablero | Área | `board_length` |
| Fichas en el pozo | Área | `pool_size` |

### Post-partida (al finalizar) — vía `/summary`

Estas gráficas requieren datos agregados de toda la partida y son más significativas al final:

| Gráfica | Tipo sugerido | Descripción |
|---|---|---|
| Comparación de costos promedio | Barras agrupadas | `avg_time_ms`, `avg_nodes`, `avg_evals`, `avg_depth` |
| Tiempo acumulado durante la partida | Área | Costo total de cómputo por estrategia |
| Nodos acumulados | Área | Espacio de búsqueda total explorado |
| Distribución de profundidades | Histograma | Frecuencia de cada nivel de búsqueda alcanzado |
| Radar multidimensional | Spider/Radar | Comparación normalizada en 5 ejes |
| Balance de pips final | Barras | Pips en mano al terminar (menor = mejor) |

**En el benchmark** (múltiples partidas) se añaden:

| Gráfica | Descripción |
|---|---|
| Win rate por matchup | Barras agrupadas por estrategia |
| Convergencia de ventaja en pips | Línea suavizada por partida |
| Distribución de duración de partidas | Histograma de turnos |

---

## Ejecutar las pruebas

```powershell
# Activar el entorno primero
.\.venv\Scripts\Activate.ps1

# Suite completa (96 tests)
pytest

# Solo tests unitarios (rápido, < 1s)
pytest tests/test_profiler.py tests/test_game_state.py tests/test_evaluator.py

# Solo tests de estrategias (partidas completas)
pytest tests/test_strategies.py

# Solo tests de integración de la API
pytest tests/test_api.py

# Con reporte de cobertura (requiere pytest-cov)
pip install pytest-cov
pytest --cov=app --cov-report=term-missing
```

Resultado esperado:

```
96 passed in ~3s
```

---

## Consumo desde el frontend

El frontend puede integrarse de tres formas según el caso de uso:

**1. Paso a paso controlado** — el frontend decide cuándo avanzar:
```
POST /api/game/{id}/step  →  evento JSON del turno
```

**2. Auto-play con streaming** — la API juega sola y emite eventos:
```
GET /api/game/{id}/stream?delay_ms=500  →  eventos SSE
```

**3. Benchmark completo** — comparativa entre estrategias:
```
POST /api/benchmark/run  →  eventos SSE por matchup completado
```

Luego de cualquier partida, los datos pre-formateados para las librerías de gráficas (Chart.js, Recharts, D3, etc.) están disponibles en:
```
GET /api/metrics/game/{id}/realtime   ←  durante la partida
GET /api/metrics/game/{id}/summary    ←  al finalizar
```

---

## Variables de entorno (opcional)

Sin configuración adicional la API funciona con valores por defecto. Para producción se recomienda crear un archivo `.env` y ajustar:

```env
# Puerto y host
HOST=0.0.0.0
PORT=8000

# CORS: reemplazar * con el origen real del frontend
ALLOWED_ORIGINS=https://mi-frontend.com
```

Y arrancar con:
```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
