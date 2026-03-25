#!/usr/bin/env python3
"""
Update CFB Power Ratings

Blends in-season ratings with a preseason baseline as the season progresses.
Each team's blend weight = min(games_played * 0.15, 1.0), so:
  - 1 game:  15% in-season, 85% preseason
  - 4 games: 60% in-season, 40% preseason
  - 7+ games: 100% in-season

If no preseason baseline exists, falls back to pure in-season ratings.

Usage:
    python update_ratings.py --season 2026
    python update_ratings.py --season 2026 --api-key YOUR_KEY
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from power_rating import RatingConfig, calculate_ratings, save_ratings

RATINGS_DIR = Path(__file__).parent / "historical_ratings"


def load_api_key() -> str:
    """Load API key from environment or config file."""
    api_key = os.environ.get("CFB_API_KEY", "")
    if not api_key:
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            config_vars = {}
            exec(open(config_path).read(), config_vars)
            api_key = config_vars.get("CFB_API_KEY", "")
    return api_key


def blend_with_preseason(in_season: pd.DataFrame, season: int,
                          per_game_step: float = 0.15) -> pd.DataFrame:
    """
    Blend in-season ratings with the preseason baseline.

    Teams that have fewer than min_fbs_games (and thus aren't in the
    in-season results yet) are carried over from preseason at full weight.
    Teams that do appear get blended: w = min(games * per_game_step, 1.0).
    """
    baseline_path = RATINGS_DIR / f"ratings_{season}_preseason.csv"
    if not baseline_path.exists():
        print(f"  No preseason baseline found at {baseline_path}, using pure in-season ratings")
        return in_season

    preseason = pd.read_csv(baseline_path)
    print(f"  Loaded preseason baseline ({len(preseason)} teams)")

    # Normalize team name for matching
    in_season = in_season.copy()
    in_season["_team_key"] = in_season["team"].str.strip()
    preseason["_team_key"] = preseason["team"].str.strip()

    merged = preseason.set_index("_team_key").join(
        in_season.set_index("_team_key"),
        how="outer",
        lsuffix="_pre",
        rsuffix="_cur"
    ).reset_index()

    results = []
    for _, row in merged.iterrows():
        games = row.get("games_cur") if pd.notna(row.get("games_cur")) else 0
        games = int(games) if not np.isnan(float(games)) else 0

        has_current = pd.notna(row.get("power_rating_cur"))
        has_preseason = pd.notna(row.get("power_rating_pre"))

        if not has_preseason and not has_current:
            continue

        if not has_preseason:
            # New team with no preseason baseline — use in-season as-is
            w = 1.0
        else:
            w = min(games * per_game_step, 1.0) if has_current else 0.0

        def blend(cur, pre):
            if pd.isna(cur) or not has_current:
                return pre
            if pd.isna(pre):
                return cur
            return round(w * cur + (1 - w) * pre, 1)

        power = blend(row.get("power_rating_cur"), row.get("power_rating_pre"))
        off   = blend(row.get("off_rating_cur"),   row.get("off_rating_pre"))
        deff  = blend(row.get("def_rating_cur"),   row.get("def_rating_pre"))
        srs   = blend(row.get("srs_cur"),          row.get("srs_pre"))
        epa   = blend(row.get("epa_rating_cur"),   row.get("epa_rating_pre"))

        team_name = row.get("team_cur") if pd.notna(row.get("team_cur")) else row.get("team_pre")

        record = row.get("record_cur") if (has_current and pd.notna(row.get("record_cur"))) else "Preseason"
        wins   = int(row.get("wins_cur",   0)) if has_current and pd.notna(row.get("wins_cur"))   else 0
        losses = int(row.get("losses_cur", 0)) if has_current and pd.notna(row.get("losses_cur")) else 0
        conf   = row.get("conference_cur") if pd.notna(row.get("conference_cur")) else row.get("conference_pre")

        results.append({
            "team": team_name,
            "conference": conf,
            "power_rating": power,
            "record": record,
            "wins": wins,
            "losses": losses,
            "off_rating": off,
            "def_rating": deff,
            "srs": srs,
            "epa_rating": epa,
            "games": games,
            "blend_weight": round(w, 2),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    # Reorder columns to match expected output format
    cols = ["rank", "team", "conference", "power_rating", "record", "wins", "losses",
            "off_rating", "def_rating", "srs", "epa_rating", "games", "blend_weight"]
    df = df[[c for c in cols if c in df.columns]]

    n_inseason = (df["blend_weight"] > 0).sum()
    n_preseason_only = (df["blend_weight"] == 0).sum()
    print(f"  Blend result: {n_inseason} teams blending in-season, {n_preseason_only} on preseason only")

    return df


def main():
    parser = argparse.ArgumentParser(description="Update CFB Power Ratings")
    parser.add_argument("--season", type=int, required=True,
                        help="Season year (e.g., 2026)")
    parser.add_argument("--api-key", type=str,
                        help="College Football Data API key (or set CFB_API_KEY env var)")
    parser.add_argument("--no-blend", action="store_true",
                        help="Skip preseason blend, output pure in-season ratings")

    args = parser.parse_args()

    config = RatingConfig()
    config.api_key = args.api_key or load_api_key()

    if not config.api_key:
        print("Error: API key required.")
        print("Options:")
        print("  1. Use --api-key argument")
        print("  2. Set CFB_API_KEY environment variable")
        print("  3. Create config.py with CFB_API_KEY = 'your-key'")
        print("\nGet your API key at: https://collegefootballdata.com/key")
        exit(1)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ratings update")

    try:
        # Calculate in-season ratings
        ratings = calculate_ratings(args.season, config)

        # Blend with preseason baseline unless disabled
        if not args.no_blend:
            print(f"\n  Blending with preseason baseline...")
            ratings = blend_with_preseason(ratings, args.season)

        # Save to CSV
        output_path = save_ratings(ratings, args.season, config)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise

    print(f"\nAll 1 season(s) updated successfully.")


if __name__ == "__main__":
    main()
