"""Position-specific player archetypes — 3 profiles per position group."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

PILLAR_KEYS: tuple[str, ...] = (
    "prog_passe",
    "ousadia",
    "construcao",
    "conducao",
    "penetracao",
)

PILLAR_METRICS: dict[str, tuple[str, ...]] = {
    "prog_passe": ("impact_passes_p90", "impact_per_pass", "positive_dxt_pct"),
    "ousadia": ("risk_pass_pct", "aggression_aip_p90"),
    "construcao": ("construction_aip_p90",),
    "conducao": (
        "carry_impact_passes_p90",
        "carry_dxt_per_pass",
        "carry_threat_carry_pct",
        "carry_positive_dxt_pct",
    ),
    "penetracao": (
        "carry_carries_impact_to_box_p90",
        "carry_dribbles_final_third_p90",
        "aggression_aip_p90",
    ),
}

PILLAR_LABELS: dict[str, str] = {
    # Threat / value added on passes (xT, StatsBomb OBV-style progression).
    "prog_passe": "Ameaça no passe",
    # Risk passes + final-third attacking impact (vertical play).
    "ousadia": "Verticalidade",
    # Build-up phase in the first 80% of the pitch (jogo de construção).
    "construcao": "Construção de jogo",
    # Ball progression via carries (Hudl/StatsBomb carrying).
    "conducao": "Condução de bola",
    # Final-third arrivals and box penetration (deep progressions).
    "penetracao": "Chegada à área",
}

# Pass-side pillars (blue) vs carry-side pillars (green) on the radar.
PILLAR_IS_CARRY: dict[str, bool] = {
    "prog_passe": False,
    "ousadia": False,
    "construcao": False,
    "conducao": True,
    "penetracao": True,
}

SUPPORTED_POSITION_GROUPS: frozenset[str] = frozenset({
    "centerbacks",
    "fullbacks",
    "midfielders",
    "wingers",
    "strikers",
})

# Prototype targets are percentile scores (0–100) within the position group.
ARCHETYPE_CATALOG: dict[str, dict[str, dict[str, Any]]] = {
    "centerbacks": {
        "vertical": {
            "label": "Vertical",
            "description": "Avança o jogo no passe com progressão e ousadia.",
            "style": "vertical",
            "icon": "fa-arrow-trend-up",
            "prototype": {
                "prog_passe": 70,
                "ousadia": 62,
                "construcao": 52,
                "conducao": 48,
                "penetracao": 28,
            },
        },
        "condutor": {
            "label": "Condutor",
            "description": "Prefere sair conduzindo e progredir com a bola no pé.",
            "style": "carry",
            "icon": "fa-person-running",
            "prototype": {
                "prog_passe": 42,
                "ousadia": 40,
                "construcao": 45,
                "conducao": 82,
                "penetracao": 32,
            },
        },
        "construtor": {
            "label": "Construtor",
            "description": "Organiza a saída de bola com construção e passe seguro.",
            "style": "build",
            "icon": "fa-diagram-project",
            "prototype": {
                "prog_passe": 58,
                "ousadia": 32,
                "construcao": 80,
                "conducao": 45,
                "penetracao": 20,
            },
        },
    },
    "fullbacks": {
        "condutor": {
            "label": "Condutor",
            "description": "Avança pela linha conduzindo com regularidade.",
            "style": "carry",
            "icon": "fa-person-running",
            "prototype": {
                "prog_passe": 48,
                "ousadia": 45,
                "construcao": 45,
                "conducao": 78,
                "penetracao": 58,
            },
        },
        "construtor": {
            "label": "Construtor",
            "description": "Organiza o lado com passe e construção, menos overlap.",
            "style": "build",
            "icon": "fa-diagram-project",
            "prototype": {
                "prog_passe": 65,
                "ousadia": 38,
                "construcao": 75,
                "conducao": 52,
                "penetracao": 48,
            },
        },
        "ofensivo": {
            "label": "Ofensivo",
            "description": "Chega alto, overlap e penetração no terço final.",
            "style": "attack",
            "icon": "fa-bolt",
            "prototype": {
                "prog_passe": 55,
                "ousadia": 55,
                "construcao": 42,
                "conducao": 70,
                "penetracao": 78,
            },
        },
    },
    "midfielders": {
        "regista": {
            "label": "Regista",
            "description": "Volante organizador: construção alta e baixo risco.",
            "style": "build",
            "icon": "fa-compass",
            "prototype": {
                "prog_passe": 58,
                "ousadia": 30,
                "construcao": 82,
                "conducao": 48,
                "penetracao": 35,
            },
        },
        "vertical": {
            "label": "Vertical",
            "description": "Meia que progride no passe com volume e ousadia.",
            "style": "vertical",
            "icon": "fa-arrow-trend-up",
            "prototype": {
                "prog_passe": 70,
                "ousadia": 68,
                "construcao": 52,
                "conducao": 55,
                "penetracao": 55,
            },
        },
        "box_to_box": {
            "label": "Box-to-box",
            "description": "Liga defesa e ataque com condução e participação ampla.",
            "style": "carry",
            "icon": "fa-arrows-left-right",
            "prototype": {
                "prog_passe": 55,
                "ousadia": 48,
                "construcao": 55,
                "conducao": 75,
                "penetracao": 48,
            },
        },
    },
    "wingers": {
        "de_toque": {
            "label": "De toque",
            "description": "Liga o jogo pelos lados, participa da construção.",
            "style": "link",
            "icon": "fa-link",
            "prototype": {
                "prog_passe": 62,
                "ousadia": 42,
                "construcao": 70,
                "conducao": 52,
                "penetracao": 48,
            },
        },
        "direto": {
            "label": "Direto",
            "description": "Linha de fundo, condução e jogo vertical pelos lados.",
            "style": "carry",
            "icon": "fa-forward",
            "prototype": {
                "prog_passe": 48,
                "ousadia": 52,
                "construcao": 38,
                "conducao": 75,
                "penetracao": 62,
            },
        },
        "criador": {
            "label": "Criador",
            "description": "Cria no último terço com passe e penetração.",
            "style": "attack",
            "icon": "fa-wand-magic-sparkles",
            "prototype": {
                "prog_passe": 68,
                "ousadia": 52,
                "construcao": 52,
                "conducao": 55,
                "penetracao": 72,
            },
        },
    },
    "strikers": {
        "referencia": {
            "label": "Referência",
            "description": "Perfil de pivô com menor envolvimento na progressão.",
            "style": "reference",
            "icon": "fa-anchor",
            "prototype": {
                "prog_passe": 42,
                "ousadia": 28,
                "construcao": 45,
                "conducao": 40,
                "penetracao": 35,
            },
        },
        "de_ligacao": {
            "label": "De ligação",
            "description": "Desce, associa e participa da construção do ataque.",
            "style": "link",
            "icon": "fa-link",
            "prototype": {
                "prog_passe": 62,
                "ousadia": 42,
                "construcao": 58,
                "conducao": 52,
                "penetracao": 52,
            },
        },
        "penetrador": {
            "label": "Penetrador",
            "description": "Chega à área e progride com condução no último terço.",
            "style": "attack",
            "icon": "fa-bullseye",
            "prototype": {
                "prog_passe": 45,
                "ousadia": 40,
                "construcao": 35,
                "conducao": 75,
                "penetracao": 78,
            },
        },
    },
}


def percentile_to_display_score(pct: float) -> float:
    """Map 0–100 within-position percentile to radar radius (3.0–9.0).

    Higher percentile (better vs peers) → larger radius. Percentile is inverted
    to rank because rank 1 is best in rank_to_display_score.
    """
    import passes_engine as pe

    bounded = max(0.0, min(100.0, float(pct)))
    rank = max(1, min(100, int(round((100.0 - bounded) / 100.0 * 99.0)) + 1))
    return float(pe.rank_to_display_score(rank, 100))


def prototype_to_display_score(value: float) -> float:
    """Map archetype prototype pillar weight (0–100) to radar radius (3.0–9.0)."""
    bounded = max(0.0, min(100.0, float(value)))
    return 3.0 + (bounded / 100.0) * 6.0


def pillar_display_scores(pillar_pct: dict[str, float] | None) -> list[float]:
    return [
        percentile_to_display_score(pillar_pct.get(key, 50.0))
        for key in PILLAR_KEYS
    ]


def prototype_display_scores(prototype: dict[str, float] | None) -> list[float]:
    return [
        prototype_to_display_score(prototype.get(key, 50.0))
        for key in PILLAR_KEYS
    ]


def archetype_prototype_for_player(player: dict) -> dict[str, float] | None:
    position_group = str(player.get("position_group") or "")
    archetype_key = player.get("player_archetype")
    if not position_group or not archetype_key:
        return None
    catalog = ARCHETYPE_CATALOG.get(position_group, {})
    spec = catalog.get(str(archetype_key))
    if not spec:
        return None
    prototype = spec.get("prototype")
    return dict(prototype) if isinstance(prototype, dict) else None


def _percentile_rank(pool_values: list[float], value: float | None) -> float | None:
    if value is None:
        return None
    clean = [float(v) for v in pool_values if v is not None and not np.isnan(v)]
    if not clean:
        return None
    return 100.0 * sum(1 for v in clean if v < float(value)) / len(clean)


def _pillar_scores(player: dict, pool: list[dict]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for pillar, metric_keys in PILLAR_METRICS.items():
        percentiles: list[float] = []
        for metric_key in metric_keys:
            pool_values = [p.get(metric_key) for p in pool]
            pct = _percentile_rank(pool_values, player.get(metric_key))
            if pct is not None:
                percentiles.append(pct)
        if percentiles:
            scores[pillar] = float(np.mean(percentiles))
    return scores


def _nearest_archetype(
    pillar_scores: dict[str, float],
    position_group: str,
) -> dict[str, Any] | None:
    catalog = ARCHETYPE_CATALOG.get(position_group)
    if not catalog or not pillar_scores:
        return None

    best_key: str | None = None
    best_dist = float("inf")
    for archetype_key, spec in catalog.items():
        prototype = spec["prototype"]
        dist = float(
            np.linalg.norm(
                [pillar_scores.get(key, 50.0) - prototype[key] for key in PILLAR_KEYS]
            )
        )
        if dist < best_dist:
            best_dist = dist
            best_key = archetype_key

    if best_key is None:
        return None

    spec = catalog[best_key]
    return {
        "player_archetype": best_key,
        "player_archetype_label": spec["label"],
        "player_archetype_description": spec["description"],
        "player_archetype_style": spec["style"],
        "player_archetype_icon": spec["icon"],
        "player_archetype_distance": round(best_dist, 2),
    }


def assign_player_archetype(player: dict, pool: list[dict]) -> dict[str, Any]:
    """Return archetype fields for one player given peers in the same position group."""
    position_group = str(player.get("position_group") or "")
    if position_group not in SUPPORTED_POSITION_GROUPS:
        return {
            "player_archetype": None,
            "player_archetype_label": None,
            "player_archetype_description": None,
            "player_archetype_style": None,
            "player_archetype_icon": None,
            "player_archetype_distance": None,
            "player_pillar_pct": {},
            "player_archetype_prototype_pct": None,
        }

    scores = _pillar_scores(player, pool)
    fields = _nearest_archetype(scores, position_group)
    if fields is None:
        return {
            "player_archetype": None,
            "player_archetype_label": None,
            "player_archetype_description": None,
            "player_archetype_style": None,
            "player_archetype_icon": None,
            "player_archetype_distance": None,
            "player_pillar_pct": {},
            "player_archetype_prototype_pct": None,
        }

    archetype_key = fields.get("player_archetype")
    prototype_pct = None
    if archetype_key and position_group in ARCHETYPE_CATALOG:
        spec = ARCHETYPE_CATALOG[position_group].get(str(archetype_key), {})
        proto = spec.get("prototype")
        if isinstance(proto, dict):
            prototype_pct = dict(proto)

    return {
        **fields,
        "player_pillar_pct": scores,
        "player_archetype_prototype_pct": prototype_pct,
    }


def attach_player_archetypes(players: list[dict]) -> list[dict]:
    """Attach position-relative archetype labels to each player dict."""
    if not players:
        return []

    by_position: dict[str, list[dict]] = defaultdict(list)
    for player in players:
        group = str(player.get("position_group") or "")
        if group in SUPPORTED_POSITION_GROUPS:
            by_position[group].append(player)

    enriched: list[dict] = []
    for player in players:
        group = str(player.get("position_group") or "")
        pool = by_position.get(group, [player])
        enriched.append({**player, **assign_player_archetype(player, pool)})
    return enriched
