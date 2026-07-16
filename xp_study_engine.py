"""Match-level xP study: per-team destination rarity with hybrid league models."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

import passes_engine as pe

STUDY_MATCH_EVENT_ID = 15526003
XP_GRID_COLS = 12
XP_GRID_ROWS = 8
OD_GRID_COLS = 6
OD_GRID_ROWS = 4
XP_SMOOTHING = 1.0
XP_BLEND_ALPHA = 0.65

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
FIRST_THIRD_LINE_X = FIELD_X / 3.0
FIRST_THIRD_BLEND_END_X = 52.0
XP_FIRST_THIRD_MIN_MULT = 0.12

XP_MODEL_MATCH_ONLY = "match_only"
XP_MODEL_MULTIPLICATIVE = "multiplicative"
XP_MODEL_HIER_DEST = "hierarchical_dest"
XP_MODEL_HIER_OD = "hierarchical_od"

XP_MODEL_LABELS: dict[str, str] = {
    XP_MODEL_MATCH_ONLY: "1 — Só partida (time)",
    XP_MODEL_MULTIPLICATIVE: "2 — Multiplicativo match^α · global^(1−α)",
    XP_MODEL_HIER_DEST: "3 — Suavização hierárquica (destino)",
    XP_MODEL_HIER_OD: "4 — Suavização hierárquica (origem→destino)",
}

XP_MODEL_COLUMNS: dict[str, str] = {
    XP_MODEL_MATCH_ONLY: "xp_match_only",
    XP_MODEL_MULTIPLICATIVE: "xp_multiplicative",
    XP_MODEL_HIER_DEST: "xp_hier_dest",
    XP_MODEL_HIER_OD: "xp_hier_od",
}


def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _cell_indices(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cols: int,
    rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_bins = np.linspace(0.0, FIELD_X, cols + 1)
    y_bins = np.linspace(0.0, FIELD_Y, rows + 1)
    x_idx = np.clip(np.digitize(x, x_bins, right=True) - 1, 0, cols - 1)
    y_idx = np.clip(np.digitize(y, y_bins, right=True) - 1, 0, rows - 1)
    return x_idx, y_idx


def _dest_cell_indices(x_end: np.ndarray, y_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _cell_indices(x_end, y_end, cols=XP_GRID_COLS, rows=XP_GRID_ROWS)


def _od_cell_indices(
    x_start: np.ndarray,
    y_start: np.ndarray,
    x_end: np.ndarray,
    y_end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ox, oy = _cell_indices(x_start, y_start, cols=OD_GRID_COLS, rows=OD_GRID_ROWS)
    dx, dy = _cell_indices(x_end, y_end, cols=OD_GRID_COLS, rows=OD_GRID_ROWS)
    return ox, oy, dx, dy


def _first_third_multiplier_vec(x_end: np.ndarray) -> np.ndarray:
    """Down-weight destinations in the defensive third (e.g. CB → GK)."""
    x = np.asarray(x_end, dtype=float)
    mult = np.ones(len(x), dtype=float)
    deep = x <= FIRST_THIRD_LINE_X
    mult[deep] = XP_FIRST_THIRD_MIN_MULT
    blend = (x > FIRST_THIRD_LINE_X) & (x < FIRST_THIRD_BLEND_END_X)
    if blend.any():
        t = (x[blend] - FIRST_THIRD_LINE_X) / (FIRST_THIRD_BLEND_END_X - FIRST_THIRD_LINE_X)
        mult[blend] = XP_FIRST_THIRD_MIN_MULT + (1.0 - XP_FIRST_THIRD_MIN_MULT) * t
    return mult


def _enrich_match_passes(frame: pd.DataFrame) -> pd.DataFrame:
    sx, sy = pe._wyscout_to_sb(frame["start_x"], frame["start_y"])
    has_end = frame["end_x"].notna() & frame["end_y"].notna()
    ex = np.full(len(frame), np.nan)
    ey = np.full(len(frame), np.nan)
    if has_end.any():
        ex[has_end.to_numpy()], ey[has_end.to_numpy()] = pe._wyscout_to_sb(
            frame.loc[has_end, "end_x"], frame.loc[has_end, "end_y"]
        )

    out = pd.DataFrame({
        "player_id": frame["player_id"].astype(str),
        "player_name": frame["player_name"].astype(str),
        "position": frame["position"].astype(str).str.strip().str.upper() if "position" in frame.columns else "CM",
        "team": np.where(
            _parse_bool_series(frame["isHome"]),
            frame["home_team"].astype(str),
            frame["away_team"].astype(str),
        ),
        "is_success": _parse_bool_series(frame["outcome"]) if "outcome" in frame.columns else False,
        "action_type": frame["eventActionType"].astype(str).str.strip().str.lower(),
        "x_start": sx,
        "y_start": sy,
        "x_end": ex,
        "y_end": ey,
        "has_end": has_end.to_numpy(),
        "event_id": frame["event_id"].astype(int),
        "home_team": frame["home_team"].astype(str),
        "away_team": frame["away_team"].astype(str),
        "match_date": frame["match_date"].astype(str),
    })
    out["is_won"] = out["is_success"].astype(bool)
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["is_restart"] = pe._restart_pass_mask(
        out["x_start"].to_numpy(dtype=float),
        out["y_start"].to_numpy(dtype=float),
        out["action_type"].astype(str).to_numpy(),
    )
    return out


def _count_destination_grid(
    passes: pd.DataFrame,
    *,
    cols: int = XP_GRID_COLS,
    rows: int = XP_GRID_ROWS,
) -> np.ndarray:
    count_grid = np.zeros((rows, cols), dtype=float)
    if passes is None or passes.empty:
        return count_grid

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return count_grid

    x_idx, y_idx = _cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
        cols=cols,
        rows=rows,
    )
    for ix, iy in zip(x_idx, y_idx):
        count_grid[iy, ix] += 1.0
    return count_grid


def _count_od_tensor(passes: pd.DataFrame) -> np.ndarray:
    tensor = np.zeros((OD_GRID_ROWS, OD_GRID_COLS, OD_GRID_ROWS, OD_GRID_COLS), dtype=float)
    if passes is None or passes.empty:
        return tensor

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return tensor

    ox, oy, dx, dy = _od_cell_indices(
        completed["x_start"].to_numpy(dtype=float),
        completed["y_start"].to_numpy(dtype=float),
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
    )
    for oxi, oyi, dxi, dyi in zip(ox, oy, dx, dy):
        tensor[oyi, oxi, dyi, dxi] += 1.0
    return tensor


def _counts_to_xp_grid(
    count_grid: np.ndarray,
    *,
    smoothing: float = XP_SMOOTHING,
) -> np.ndarray:
    rows, cols = count_grid.shape
    xp_grid = np.ones((rows, cols), dtype=float)
    total = float(count_grid.sum())
    num_cells = rows * cols
    denom = total + smoothing * num_cells
    for iy in range(rows):
        for ix in range(cols):
            smoothed_count = count_grid[iy, ix] + smoothing
            freq = smoothed_count / denom
            xp_grid[iy, ix] = 1.0 / freq

    weights = count_grid + smoothing
    mean_weight = float(np.average(xp_grid, weights=weights))
    if mean_weight > 0:
        xp_grid /= mean_weight
    return xp_grid


def _blend_count_grid(
    match_count: np.ndarray,
    league_count_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_count + (1.0 - alpha) * league_count_per_match


def _blend_od_tensor(
    match_tensor: np.ndarray,
    league_tensor_per_match: np.ndarray,
    *,
    alpha: float = XP_BLEND_ALPHA,
) -> np.ndarray:
    return alpha * match_tensor + (1.0 - alpha) * league_tensor_per_match


def _od_counts_to_lookup(tensor: np.ndarray) -> np.ndarray:
    """Convert origin→destination counts to rarity lookup (same smoothing logic)."""
    total = float(tensor.sum())
    num_cells = tensor.size
    denom = total + XP_SMOOTHING * num_cells
    lookup = np.ones_like(tensor, dtype=float)
    for oyi in range(OD_GRID_ROWS):
        for oxi in range(OD_GRID_COLS):
            for dyi in range(OD_GRID_ROWS):
                for dxi in range(OD_GRID_COLS):
                    smoothed = tensor[oyi, oxi, dyi, dxi] + XP_SMOOTHING
                    lookup[oyi, oxi, dyi, dxi] = 1.0 / (smoothed / denom)

    weights = tensor + XP_SMOOTHING
    mean_weight = float(np.average(lookup, weights=weights))
    if mean_weight > 0:
        lookup /= mean_weight
    return lookup


def build_destination_xp_grid(passes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    count_grid = _count_destination_grid(passes)
    return _counts_to_xp_grid(count_grid), count_grid


def build_team_xp_surfaces(
    passes: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    xp_grids: dict[str, np.ndarray] = {}
    count_grids: dict[str, np.ndarray] = {}
    if passes is None or passes.empty:
        return xp_grids, count_grids

    for team, grp in passes.groupby("team", sort=False):
        team_name = str(team)
        xp_grids[team_name], count_grids[team_name] = build_destination_xp_grid(grp)
    return xp_grids, count_grids


@functools.lru_cache(maxsize=1)
def _league_completed_passes() -> pd.DataFrame:
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return pd.DataFrame()
    passes = _enrich_match_passes(frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None or passes.empty:
        return pd.DataFrame()
    return passes[passes["is_won"] & passes["has_end"]].copy()


@functools.lru_cache(maxsize=1)
def _league_reference_surfaces() -> dict[str, np.ndarray | float | int]:
    completed = _league_completed_passes()
    if completed.empty:
        empty_dest = np.zeros((XP_GRID_ROWS, XP_GRID_COLS), dtype=float)
        empty_od = np.zeros((OD_GRID_ROWS, OD_GRID_COLS, OD_GRID_ROWS, OD_GRID_COLS), dtype=float)
        return {
            "dest_count": empty_dest,
            "dest_count_per_match": empty_dest,
            "dest_xp": _counts_to_xp_grid(empty_dest),
            "od_count": empty_od,
            "od_count_per_match": empty_od,
            "od_lookup": _od_counts_to_lookup(empty_od),
            "num_matches": 1,
        }

    dest_count = _count_destination_grid(completed)
    od_count = _count_od_tensor(completed)
    num_matches = max(int(completed["event_id"].nunique()), 1)
    dest_per_match = dest_count / num_matches
    od_per_match = od_count / num_matches

    return {
        "dest_count": dest_count,
        "dest_count_per_match": dest_per_match,
        "dest_xp": _counts_to_xp_grid(dest_count),
        "od_count": od_count,
        "od_count_per_match": od_per_match,
        "od_lookup": _od_counts_to_lookup(od_count),
        "num_matches": num_matches,
    }


def _assign_all_xp_models(
    passes: pd.DataFrame,
    *,
    xp_grids_by_team: dict[str, np.ndarray],
    count_grids_by_team: dict[str, np.ndarray],
    league: dict[str, np.ndarray | float | int],
    alpha: float = XP_BLEND_ALPHA,
) -> pd.DataFrame:
    out = passes.copy()
    for col in XP_MODEL_COLUMNS.values():
        out[col] = 0.0
    out["xp_zone_mult"] = 0.0
    out["dest_ix"] = -1
    out["dest_iy"] = -1

    mask = out["is_won"] & out["has_end"]
    if not mask.any():
        return out

    sub = out.loc[mask]
    x_idx, y_idx = _dest_cell_indices(
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
    )
    ox, oy, odx, ody = _od_cell_indices(
        sub["x_start"].to_numpy(dtype=float),
        sub["y_start"].to_numpy(dtype=float),
        sub["x_end"].to_numpy(dtype=float),
        sub["y_end"].to_numpy(dtype=float),
    )
    zone_mult = _first_third_multiplier_vec(sub["x_end"].to_numpy(dtype=float))

    league_dest_xp = league["dest_xp"]  # type: ignore[index]
    league_dest_per_match = league["dest_count_per_match"]  # type: ignore[index]
    league_od_per_match = league["od_count_per_match"]  # type: ignore[index]

    hier_dest_xp_by_team: dict[str, np.ndarray] = {}
    hier_od_lookup_by_team: dict[str, np.ndarray] = {}
    for team, count_grid in count_grids_by_team.items():
        blended_dest = _blend_count_grid(count_grid, league_dest_per_match, alpha=alpha)
        hier_dest_xp_by_team[team] = _counts_to_xp_grid(blended_dest)
        team_od = _count_od_tensor(passes[passes["team"].astype(str) == team])
        blended_od = _blend_od_tensor(team_od, league_od_per_match, alpha=alpha)
        hier_od_lookup_by_team[team] = _od_counts_to_lookup(blended_od)

    match_vals = np.zeros(len(sub), dtype=float)
    mult_vals = np.zeros(len(sub), dtype=float)
    hier_dest_vals = np.zeros(len(sub), dtype=float)
    hier_od_vals = np.zeros(len(sub), dtype=float)

    for i, (team, iy, ix, oyi, oxi, dyi, dxi) in enumerate(
        zip(sub["team"].astype(str), y_idx, x_idx, oy, ox, ody, odx)
    ):
        team_grid = xp_grids_by_team.get(team)
        match_xp = float(team_grid[iy, ix]) if team_grid is not None else 1.0
        global_xp = float(league_dest_xp[iy, ix])
        match_vals[i] = match_xp
        mult_vals[i] = (match_xp ** alpha) * (global_xp ** (1.0 - alpha))

        hier_grid = hier_dest_xp_by_team.get(team)
        hier_dest_vals[i] = float(hier_grid[iy, ix]) if hier_grid is not None else 1.0

        od_lookup = hier_od_lookup_by_team.get(team)
        hier_od_vals[i] = float(od_lookup[oyi, oxi, dyi, dxi]) if od_lookup is not None else 1.0

    out.loc[mask, "dest_ix"] = x_idx
    out.loc[mask, "dest_iy"] = y_idx
    out.loc[mask, "xp_zone_mult"] = zone_mult
    out.loc[mask, "xp_match_only"] = match_vals * zone_mult
    out.loc[mask, "xp_multiplicative"] = mult_vals * zone_mult
    out.loc[mask, "xp_hier_dest"] = hier_dest_vals * zone_mult
    out.loc[mask, "xp_hier_od"] = hier_od_vals * zone_mult
    return out


def rank_players_by_xp(
    passes: pd.DataFrame,
    *,
    model: str = XP_MODEL_MATCH_ONLY,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_match_only")
    if passes is None or passes.empty or col not in passes.columns:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        rows.append({
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "position": str(grp["position"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
            "xp_total": float(grp[col].sum()),
            "xp_per_pass": float(grp[col].mean()),
            "xp_max_pass": float(grp[col].max()),
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(["xp_total", "xp_per_pass"], ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def build_model_comparison_table(passes: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side player totals for all xP models."""
    if passes is None or passes.empty:
        return pd.DataFrame()

    scored = passes[passes["is_won"] & passes["has_end"]].copy()
    if scored.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for pid, grp in scored.groupby("player_id", sort=False):
        row = {
            "player_id": str(pid),
            "player_name": str(grp["player_name"].iloc[0]),
            "team": str(grp["team"].mode().iloc[0] if not grp["team"].mode().empty else grp["team"].iloc[0]),
            "passes_completed": int(len(grp)),
        }
        for model, col in XP_MODEL_COLUMNS.items():
            row[f"xp_{model}"] = float(grp[col].sum())
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    for model in XP_MODEL_COLUMNS:
        col = f"xp_{model}"
        table[f"rank_{model}"] = table[col].rank(ascending=False, method="min").astype(int)

    table = table.sort_values("xp_match_only", ascending=False).reset_index(drop=True)
    return table


def top_xp_passes_for_player(
    passes: pd.DataFrame,
    player_id: str,
    *,
    n: int = 5,
    model: str = XP_MODEL_MATCH_ONLY,
) -> pd.DataFrame:
    col = XP_MODEL_COLUMNS.get(model, "xp_match_only")
    subset = passes[
        (passes["player_id"].astype(str) == str(player_id))
        & passes["is_won"]
        & passes["has_end"]
    ].copy()
    if subset.empty or col not in subset.columns:
        return subset
    subset = subset.assign(xp_value=subset[col])
    return subset.sort_values("xp_value", ascending=False).head(n).reset_index(drop=True)


def match_label(meta: dict) -> str:
    return f"{meta['home_team']} vs {meta['away_team']}"


def team_surface_for_player(
    xp_grids_by_team: dict[str, np.ndarray],
    count_grids_by_team: dict[str, np.ndarray],
    *,
    team: str,
) -> tuple[np.ndarray, np.ndarray]:
    empty_xp = np.ones((XP_GRID_ROWS, XP_GRID_COLS), dtype=float)
    empty_count = np.zeros((XP_GRID_ROWS, XP_GRID_COLS), dtype=float)
    return (
        xp_grids_by_team.get(team, empty_xp),
        count_grids_by_team.get(team, empty_count),
    )


def normalize_xp_model(model: str | None) -> str:
    key = str(model or XP_MODEL_MATCH_ONLY).strip().lower()
    return key if key in XP_MODEL_COLUMNS else XP_MODEL_MATCH_ONLY


@functools.lru_cache(maxsize=4)
def load_study_match_bundle(event_id: int = STUDY_MATCH_EVENT_ID) -> dict:
    """Load one match, build all xP models, and return passes + rankings."""
    empty = {
        "passes": pd.DataFrame(),
        "xp_grids_by_team": {},
        "count_grids_by_team": {},
        "league": {},
        "rankings_by_model": {},
        "comparison": pd.DataFrame(),
        "meta": {},
    }
    frame = pe._load_season_pass_frame()
    if frame.empty:
        return empty

    match_frame = frame[frame["event_id"].astype(int) == int(event_id)].copy()
    if match_frame.empty:
        return empty

    passes = _enrich_match_passes(match_frame)
    passes = pe.filter_live_ball_passes(passes)
    if passes is None:
        passes = pd.DataFrame()

    league = _league_reference_surfaces()
    xp_grids_by_team, count_grids_by_team = build_team_xp_surfaces(passes)
    if not passes.empty:
        passes = _assign_all_xp_models(
            passes,
            xp_grids_by_team=xp_grids_by_team,
            count_grids_by_team=count_grids_by_team,
            league=league,
        )

    first = match_frame.iloc[0]
    home_team = str(first["home_team"])
    away_team = str(first["away_team"])
    meta = {
        "event_id": int(event_id),
        "home_team": home_team,
        "away_team": away_team,
        "match_date": str(first["match_date"])[:10],
        "pass_events": int(len(match_frame)),
        "live_ball_passes": int(len(passes)),
        "completed_passes": int((passes["is_won"] & passes["has_end"]).sum()) if not passes.empty else 0,
        "players": int(passes["player_id"].nunique()) if not passes.empty else 0,
        "home_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == home_team)).sum()
        ) if not passes.empty else 0,
        "away_completed": int(
            (passes["is_won"] & passes["has_end"] & (passes["team"] == away_team)).sum()
        ) if not passes.empty else 0,
        "league_matches": int(league.get("num_matches", 0)),
        "blend_alpha": XP_BLEND_ALPHA,
    }

    rankings_by_model = {
        model: rank_players_by_xp(passes, model=model)
        for model in XP_MODEL_COLUMNS
    }
    comparison = build_model_comparison_table(passes)

    return {
        "passes": passes,
        "xp_grids_by_team": xp_grids_by_team,
        "count_grids_by_team": count_grids_by_team,
        "league": league,
        "rankings_by_model": rankings_by_model,
        "comparison": comparison,
        "meta": meta,
    }
