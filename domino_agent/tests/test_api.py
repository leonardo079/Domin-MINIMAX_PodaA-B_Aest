"""
tests/test_api.py — Tests de integración de los endpoints FastAPI.

Usa httpx.AsyncClient + ASGITransport para probar la API en memoria
sin levantar un servidor real.
"""
import pytest
from httpx import AsyncClient
from app.api import game_manager
from app.core.game_state import OrientedTile, Tile


# ── Root ───────────────────────────────────────────────────────────────────────

async def test_root(client: AsyncClient):
    r = await client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "Dominó AI API"
    assert "endpoints" in data


# ── GET /api/metrics/strategies ────────────────────────────────────────────────

async def test_list_strategies(client: AsyncClient):
    r = await client.get("/api/metrics/strategies")
    assert r.status_code == 200
    data = r.json()
    assert "strategies" in data
    names = [s["name"] for s in data["strategies"]]
    for expected in ("random", "manhattan", "euclidean", "astar", "hybrid"):
        assert expected in names


# ── POST /api/game/new ─────────────────────────────────────────────────────────

async def test_new_game(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random",
        "strategy_b": "random"
    })
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["strategy_a"] == "random"
    assert data["strategy_b"] == "random"
    assert data["status"] == "active"
    return data["session_id"]


async def test_new_game_invalid_strategy(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "nonexistent",
        "strategy_b": "random"
    })
    assert r.status_code == 422  # Pydantic validation error


# ── GET /api/game/ ─────────────────────────────────────────────────────────────

async def test_list_games(client: AsyncClient):
    # Crear una sesión primero
    await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    r = await client.get("/api/game/")
    assert r.status_code == 200
    assert "sessions" in r.json()


# ── GET /api/game/{id} ─────────────────────────────────────────────────────────

async def test_get_game_state(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    r2 = await client.get(f"/api/game/{sid}")
    assert r2.status_code == 200
    state = r2.json()
    assert state["session_id"] == sid
    assert "board_str" in state
    assert "pool_size" in state


async def test_get_game_not_found(client: AsyncClient):
    r = await client.get("/api/game/nonexistent-id")
    assert r.status_code == 404


# ── POST /api/game/{id}/step ───────────────────────────────────────────────────

async def test_step_game(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    r2 = await client.post(f"/api/game/{sid}/step")
    assert r2.status_code == 200
    event = r2.json()
    assert event["type"] in ("turn", "game_over")
    assert "turn" in event


async def test_step_until_terminal(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    for _ in range(200):
        r2 = await client.post(f"/api/game/{sid}/step")
        assert r2.status_code == 200
        event = r2.json()
        if event.get("is_terminal") or event["type"] == "game_over":
            break
    else:
        pytest.fail("La partida no terminó en 200 turnos")


async def test_step_finished_game_returns_400(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    # Terminar la partida
    for _ in range(200):
        r2 = await client.post(f"/api/game/{sid}/step")
        event = r2.json()
        if event.get("is_terminal") or event["type"] == "game_over":
            break

    # Intentar un turno más
    r3 = await client.post(f"/api/game/{sid}/step")
    assert r3.status_code == 400


# ── GET /api/game/{id}/history ─────────────────────────────────────────────────

async def test_game_history(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    # Jugar 3 turnos
    for _ in range(3):
        await client.post(f"/api/game/{sid}/step")

    r2 = await client.get(f"/api/game/{sid}/history")
    assert r2.status_code == 200
    hist = r2.json()
    assert hist["session_id"] == sid
    assert isinstance(hist["history"], list)


# ── DELETE /api/game/{id} ──────────────────────────────────────────────────────

async def test_delete_game(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    r2 = await client.delete(f"/api/game/{sid}")
    assert r2.status_code == 200
    assert r2.json()["deleted"] == sid

    r3 = await client.get(f"/api/game/{sid}")
    assert r3.status_code == 404


# ── GET /api/metrics/game/{id}/realtime ───────────────────────────────────────

async def test_realtime_metrics(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "manhattan", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    # Jugar unos turnos para tener datos
    for _ in range(5):
        await client.post(f"/api/game/{sid}/step")

    r2 = await client.get(f"/api/metrics/game/{sid}/realtime")
    assert r2.status_code == 200
    data = r2.json()
    assert "realtime_charts" in data
    assert "game_progress_charts" in data
    assert "time_ms" in data["realtime_charts"]
    assert "nodes_expanded" in data["realtime_charts"]


# ── GET /api/metrics/game/{id}/summary ────────────────────────────────────────

async def test_endgame_summary(client: AsyncClient):
    r = await client.post("/api/game/new", json={
        "strategy_a": "random", "strategy_b": "random"
    })
    sid = r.json()["session_id"]

    # Terminar la partida
    for _ in range(200):
        r2 = await client.post(f"/api/game/{sid}/step")
        event = r2.json()
        if event.get("is_terminal") or event["type"] == "game_over":
            break

    r3 = await client.get(f"/api/metrics/game/{sid}/summary")
    assert r3.status_code == 200
    data = r3.json()
    assert "endgame_charts" in data
    assert "cost_comparison" in data["endgame_charts"]
    assert "radar" in data["endgame_charts"]
    assert data["status"] == "finished"


# ── GET /api/benchmark/matchups ────────────────────────────────────────────────

async def test_benchmark_matchups(client: AsyncClient):
    r = await client.get("/api/benchmark/matchups")
    assert r.status_code == 200
    data = r.json()
    assert "matchups" in data
    assert len(data["matchups"]) == 5


# ── POST /api/game/{id}/human-pass ───────────────────────────────────────────

async def test_human_pass_draws_from_pool_when_no_valid_moves(client: AsyncClient):
    created = await client.post("/api/game/new", json={
        "strategy_a": "random",
        "game_mode": "agent_vs_human",
    })
    assert created.status_code == 200
    sid = created.json()["session_id"]

    session = game_manager.get_session(sid)
    assert session is not None
    session.state.current_player = 1
    session.status = "waiting_human"
    session.state.board = [OrientedTile(6, 6)]
    session.state.left_end = 6
    session.state.right_end = 6
    session.state.opponent_hand = [Tile(1, 2)]
    session.state.pool = [Tile(3, 4), Tile(6, 1)]

    action = await client.post(f"/api/game/{sid}/human-pass")
    assert action.status_code == 200
    event = action.json()
    assert event["move"] == "draw_from_pool"
    assert event["drew_from_pool"] is True

    snapshot_res = await client.get(f"/api/game/{sid}")
    assert snapshot_res.status_code == 200
    snapshot = snapshot_res.json()
    assert snapshot["status"] == "waiting_human"
    assert snapshot["pool_size"] == 0
    assert len(snapshot["human_valid_moves"]) > 0


async def test_human_pass_succeeds_when_no_valid_moves_and_no_pool(client: AsyncClient):
    created = await client.post("/api/game/new", json={
        "strategy_a": "random",
        "game_mode": "agent_vs_human",
    })
    assert created.status_code == 200
    sid = created.json()["session_id"]

    session = game_manager.get_session(sid)
    assert session is not None
    session.state.current_player = 1
    session.status = "waiting_human"
    session.state.board = [OrientedTile(6, 6)]
    session.state.left_end = 6
    session.state.right_end = 6
    session.state.opponent_hand = [Tile(1, 2)]
    session.state.pool = []

    action = await client.post(f"/api/game/{sid}/human-pass")
    assert action.status_code == 200
    event = action.json()
    assert event["move"] == "pass"
    assert event["player"] == 1
