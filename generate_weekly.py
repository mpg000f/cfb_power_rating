#!/usr/bin/env python3
"""
Generate week-by-week CFB power rating snapshots for a season.

Fetches the season's games and plays ONCE, then replays the rating pipeline
"as of" each completed week, writing one CSV per week to:
    historical_ratings/weekly/{season}/week_{NN}.csv

Each week's pure in-season ratings are blended with the season's preseason
baseline (ratings_{season}_preseason.csv) using the same games-based fade as
the live model: w = min(games * 0.15, 1.0). Early weeks therefore lean on the
preseason prior and converge to pure in-season by ~week 7, giving full week-1
coverage. Without a preseason baseline it falls back to pure in-season (and
early weeks are skipped for having too few rated teams).

Usage:
    python generate_weekly.py --season 2024
    python generate_weekly.py --season 2024 --min-teams 50
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from power_rating import (RatingConfig, calculate_ratings, fetch_games,
                          fetch_all_plays, _effective_week)
from update_ratings import blend_with_preseason

WEEKLY_DIR = Path(__file__).parent / "historical_ratings" / "weekly"


def load_api_key() -> str:
    key = os.environ.get("CFB_API_KEY", "")
    if not key:
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            cfg = {}
            exec(open(config_path).read(), cfg)
            key = cfg.get("CFB_API_KEY", "")
    return key


def completed_weeks(games: pd.DataFrame) -> list:
    """Weeks that have at least one played (scored) game."""
    played = games[games.get("homePoints").notna() & games.get("awayPoints").notna()]
    weeks = sorted(int(w) for w in _effective_week(played).dropna().unique())
    return weeks


def main():
    parser = argparse.ArgumentParser(description="Generate weekly CFB rating snapshots")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--min-teams", type=int, default=50,
                        help="Skip weeks that rate fewer than this many teams")
    args = parser.parse_args()

    config = RatingConfig()
    config.api_key = args.api_key or load_api_key()
    if not config.api_key:
        print("Error: no CFB API key (set CFB_API_KEY or config.py)")
        return

    # Let teams enter the in-season rating as soon as they have any FBS games;
    # the preseason blend weight (games * 0.15) damps small samples heavily, so
    # early ratings stay anchored to the preseason prior instead of exploding.
    config.min_fbs_games = 1

    baseline = WEEKLY_DIR.parent / f"ratings_{args.season}_preseason.csv"
    if baseline.exists():
        print(f"Using preseason baseline: {baseline.name}")
    else:
        print(f"No preseason baseline for {args.season}; early weeks fall back to pure in-season")

    print(f"Fetching {args.season} season data once...")
    games = fetch_games(args.season, config)
    plays = fetch_all_plays(args.season, config)

    weeks = completed_weeks(games)
    print(f"Completed weeks: {weeks}")

    out_dir = WEEKLY_DIR / str(args.season)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for wk in weeks:
        in_season = calculate_ratings(args.season, config, through_week=wk,
                                      prefetched=(games, plays))
        ratings = blend_with_preseason(in_season, args.season)
        if len(ratings) < args.min_teams:
            print(f"  Week {wk}: only {len(ratings)} teams rated, skipping")
            continue
        path = out_dir / f"week_{wk:02d}.csv"
        ratings.to_csv(path, index=False)
        saved.append(wk)
        print(f"  Week {wk}: saved {len(ratings)} teams -> {path.name}")

    print(f"\n{args.season}: saved {len(saved)} weekly snapshots ({saved})")


if __name__ == "__main__":
    main()
