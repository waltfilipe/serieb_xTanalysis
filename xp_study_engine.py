"""Match-level xP study: per-team destination rarity as pass value."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

import passes_engine as pe

STUDY_MATCH_EVENT_ID = 15526003
XP_GRID_COLS = 12
XP_GRID_ROWS = 8
XP_SMOOTHING = 1.0

FIELD_X = pe.FIELD_X
FIELD_Y = pe.FIELD_Y
FIRST_THIRD_LINE_X = FIELD_X / 3.0
FIRST_THIRD_BLEND_END_X = 52.0
XP_FIRST_THIRD_MIN_MULT = 0.12


def _parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "successful"})


def _dest_cell_indices(x_end: np.ndarray, y_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_bins = np.linspace(0.0, FIELD_X, XP_GRID_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, XP_GRID_ROWS + 1)
    x_idx = np.clip(np.digitize(x_end, x_bins, right=True) - 1, 0, XP_GRID_COLS - 1)
    y_idx = np.clip(np.digitize(y_end, y_bins, right=True) - 1, 0, XP_GRID_ROWS - 1)
    return x_idx, y_idx


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


def build_destination_xp_grid(passes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (xp_grid rows×cols, count_grid) from one team's completed destinations."""
    xp_grid = np.ones((XP_GRID_ROWS, XP_GRID_COLS), dtype=float)
    count_grid = np.zeros((XP_GRID_ROWS, XP_GRID_COLS), dtype=float)

    if passes is None or passes.empty:
        return xp_grid, count_grid

    completed = passes[passes["is_won"] & passes["has_end"]]
    if completed.empty:
        return xp_grid, count_grid

    x_idx, y_idx = _dest_cell_indices(
        completed["x_end"].to_numpy(dtype=float),
        completed["y_end"].to_numpy(dtype=float),
    )
    for ix, iy in zip(x_idx, y_idx):
        count_grid[iy, ix] += 1.0

    total = float(count_grid.sum())
    num_cells = XP_GRID_ROWS * XP_GRID_COLS
    denom = total + XP_SMOOTHING * num_cells
    for iy in range(XP_GRID_ROWS):
        for ix in range(XP_GRID_COLS):
            smoothed_count = count_grid[iy, ix] + XP_SMOOTHING
            freq = smoothed_count / denom
            xp_grid[iy, ix] = 1.0 / freq

    mean_weight = float(np.average(xp_grid, weights=count_grid + XP_SMOOTHING))
    if mean_weight > 0:
        xp_grid /= mean_weight
    return xp_grid, count_grid


def build_team_xp_surfaces(
    passes: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """One destination-rarity surface per team in the match."""
    xp_grids: dict[str, np.ndarray] = {}
    count_grids: dict[str, np.ndarray] = {}
    if passes is None or passes.empty:
        return xp_grids, count_grids

    for team, grp in passes.groupby("team", sort=False):
        team_name = str(team)
        xp_grids[team_name], count_grids[team_name] = build_destination_xp_grid(grp)
    return xp_grids, count_grids


def assign_pass_xp(
    passes: pd.DataFrame,
    xp_grids_by_team: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Attach per-pass xP from team-specific destination rarity + 1st-third penalty."""
    out = passes.copy()
    out["xp_value"] = 0.0
    out["xp_base"] = 0.0
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
    zone_mult = _first_third_multiplier_vec(sub["x_end"].to_numpy(dtype=float))

    base_vals = np.zeros(len(sub), dtype=float)
    for i, (team, iy, ix) in enumerate(zip(sub["team"].astype(str), y_idx, x_idx)):
        grid = xp_grids_by_team.get(team)
        if grid is None:
            base_vals[i] = 1.0
        else:
            base_vals[i] = float(grid[iy, ix])

    xp_vals = base_vals * zone_mult
    out.loc[mask, "dest_ix"] = x_idx
    out.loc[mask, "dest_iy"] = y_idx
    out.loc[mask, "xp_base"] = base_vals
    out.loc[mask, "xp_zone_mult"] = zone_mult
    out.loc[mask, "xp_value"] = xp_vals
    return out


def rank_players_by_xp(passes: pd.DataFrame) -> pd.DataFrame:
    """Rank players by total xP in the match."""
    if passes is None or passes.empty:
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
            "xp_total": float(grp["xp_value"].sum()),
            "xp_per_pass": float(grp["xp_value"].mean()),
            "xp_max_pass": float(grp["xp_value"].max()),
        })

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    ranking = ranking.sort_values(["xp_total", "xp_per_pass"], ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def top_xp_passes_for_player(passes: pd.DataFrame, player_id: str, *, n: int = 5) -> pd.DataFrame:
    subset = passes[
        (passes["player_id"].astype(str) == str(player_id))
        & passes["is_won"]
        & passes["has_end"]
    ].copy()
    if subset.empty:
        return subset
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


@functools.lru_cache(maxsize=4)
def load_study_match_bundle(event_id: int = STUDY_MATCH_EVENT_ID) -> dict:
    """Load one match, build per-team xP surfaces, and return passes + rankings."""
    empty = {
        "passes": pd.DataFrame(),
        "xp_grids_by_team": {},
        "count_grids_by_team": {},
        "ranking": pd.DataFrame(),
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

    xp_grids_by_team, count_grids_by_team = build_team_xp_surfaces(passes)
    if not passes.empty:
        passes = assign_pass_xp(passes, xp_grids_by_team)

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
    }

    ranking = rank_players_by_xp(passes)
    return {
        "passes": passes,
        "xp_grids_by_team": xp_grids_by_team,
        "count_grids_by_team": count_grids_by_team,
        "ranking": ranking,
        "meta": meta,
    }
