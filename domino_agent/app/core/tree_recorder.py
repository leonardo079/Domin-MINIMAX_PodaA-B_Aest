"""
tree_recorder.py — Registra el árbol de búsqueda generado por cada estrategia.

Permite visualizar en tiempo real:
  - Árbol Minimax con poda α-β (Manhattan, Euclidean, Hybrid)
  - Árbol A* best-first (AStarStrategy)
  - Vista simplificada (RandomStrategy)

Cada nodo tiene:
  id, parent_id, depth, node_type, move, alpha, beta, value, pruned

node_type valores:
  "MAX"           — turno maximizador en Minimax
  "MIN"           — turno minimizador en Minimax
  "PRUNED"        — placeholder de ramas cortadas por α-β
  "ROOT"          — nodo raíz de decisión (creado en decide())
  "ASTAR"         — nodo en búsqueda A*; alpha=f, beta=h, value=g
  "ASTAR_PHASE"   — subtítulo fase de ranking A* en Hybrid
  "ASTAR_RANK"    — candidato clasificado en la fase A* del Hybrid
  "MINIMAX_PHASE" — subtítulo fase Minimax en Hybrid
  "RANDOM"        — jugada disponible en estrategia aleatoria
"""
from __future__ import annotations
from typing import Optional


def _safe(v: Optional[float]) -> Optional[float]:
    """Convierte floats para JSON: elimina inf/-inf y redondea."""
    if v is None:
        return None
    if v != v:          # NaN
        return None
    if v == float('inf') or v == float('-inf'):
        return None
    return round(v, 4)


class TreeRecorder:
    """Registra nodos del árbol de búsqueda de forma compacta."""

    MAX_NODES: int = 400

    def __init__(self):
        self._nodes: list[dict] = []
        self._counter: int = 0
        self.truncated: bool = False

    # ── API pública ───────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._nodes = []
        self._counter = 0
        self.truncated = False

    def add_node(
        self,
        parent_id: Optional[int],
        depth: int,
        node_type: str,
        move: str,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        value: Optional[float] = None,
        pruned: bool = False,
    ) -> int:
        """
        Añade un nodo. Retorna el ID asignado, o -1 si se superó MAX_NODES.
        """
        if len(self._nodes) >= self.MAX_NODES:
            self.truncated = True
            return -1

        node_id = self._counter
        self._counter += 1
        self._nodes.append({
            "id":        node_id,
            "parent_id": parent_id,
            "depth":     depth,
            "node_type": node_type,
            "move":      move,
            "alpha":     _safe(alpha),
            "beta":      _safe(beta),
            "value":     _safe(value),
            "pruned":    pruned,
        })
        return node_id

    def update_value(self, node_id: int, value: float) -> None:
        if 0 <= node_id < len(self._nodes):
            self._nodes[node_id]["value"] = _safe(value)

    def to_dict(self) -> dict:
        return {
            "total_nodes": len(self._nodes),
            "truncated":   self.truncated,
            "nodes":       list(self._nodes),
        }
