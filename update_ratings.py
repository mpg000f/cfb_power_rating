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
import requests
from pathlib import Path
from datetime import datetime

from power_rating import (RatingConfig, calculate_ratings, save_ratings,
                          fetch_games, fetch_all_plays)

RATINGS_DIR = Path(__file__).parent / "historical_ratings"

# Prior anchoring. From 2026 the preseason rating goes *inside* the fit as a
# ridge target rather than being averaged in afterwards, so a team with one
# game is pulled toward its own prior instead of toward league average. That
# is what makes a week-1 result mean something: with the old approach a slate
# where no two teams share an opponent has exactly one self-consistent answer
# -- everyone is average -- so blending only shrank teams toward the mean and
# actually downgraded week-1 winners.
#
# Backtested on 2024-25, predicting each week from the ratings before it:
#   2025  MAE 11.99 -> 11.41    2024  MAE 13.76 -> 13.38
# Straight-up accuracy improved in 2025 (72.3% -> 73.1%) and fell in 2024
# (69.8% -> 66.0%), so the gain is in margin accuracy, not in picking winners.
FIRST_ANCHORED_SEASON = 2026
PRIOR_WEIGHT = 2.0   # prior counts as two played games
# Cap the surprise rather than the raw margin. A flat 28 reads a 56-3 win
# over a team projected 53 points worse as underperformance; capping the
# deviation from the prior expectation limits garbage time without punishing
# a team for being good. Backtests were indifferent between cap schemes
# (within 0.1 MAE), so this is chosen on behaviour, not on fit.
RESIDUAL_CAP = 28.0


def carry_unplayed(ratings: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Add teams that have not played yet, at their preseason rating.

    calculate_ratings only returns teams with games. The old blend carried the
    rest through implicitly; anchoring skips that blend, so they have to be
    added back or half of FBS disappears from the table in September.
    """
    if priors is None or ratings is None or len(ratings) == 0:
        return ratings
    missing = priors[~priors["team"].isin(set(ratings["team"]))].copy()
    if missing.empty:
        return ratings
    missing["record"] = "Preseason"
    missing["wins"] = 0
    missing["losses"] = 0
    missing["games"] = 0
    for col in ratings.columns:
        if col not in missing.columns:
            missing[col] = pd.NA
    out = pd.concat([ratings, missing[ratings.columns]], ignore_index=True)
    out = out.sort_values("power_rating", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    print(f"  Carried {len(missing)} teams yet to play at their preseason rating")
    return out


def sp_plus_priors(season: int, baseline: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Fill in teams missing from our baseline using preseason SP+.

    Teams new to FBS have no preseason rating of ours, and left unanchored a
    single result floats them absurdly high -- North Dakota State came out 9th
    nationally off one game. SP+ covers them. It is a different scale, so it is
    mapped onto ours by least squares against the teams present in both, which
    fit at r=0.95 for 2026.
    """
    try:
        resp = requests.get("https://api.collegefootballdata.com/ratings/sp",
                            headers={"Authorization": f"Bearer {api_key}"},
                            params={"year": season}, timeout=30)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        print(f"  SP+ unavailable ({e.__class__.__name__}); new teams fall back to a low prior")
        return baseline

    sp = pd.DataFrame([{
        "team": x.get("team"),
        "sp": x.get("rating"),
        "sp_off": (x.get("offense") or {}).get("rating"),
        "sp_def": (x.get("defense") or {}).get("rating"),
    } for x in rows if x.get("team") and x.get("team") != "nationalAverages"])

    missing = sp[~sp["team"].isin(set(baseline["team"]))].dropna(subset=["sp"])
    if missing.empty:
        return baseline

    fit = baseline.merge(sp, on="team").dropna(subset=["sp"])
    if len(fit) < 30:
        print("  Too few teams to map SP+ onto our scale; skipping")
        return baseline

    added = {"team": missing["team"].tolist()}
    for ours, theirs in [("power_rating", "sp"), ("off_rating", "sp_off"),
                         ("def_rating", "sp_def")]:
        d = fit.dropna(subset=[ours, theirs])
        slope, intercept = np.polyfit(d[theirs], d[ours], 1)
        added[ours] = (slope * missing[theirs] + intercept).round(1).tolist()

    add = pd.DataFrame(added)
    for col in baseline.columns:
        if col not in add.columns:
            add[col] = pd.NA
    named = ", ".join(f"{t} {r:+.1f}" for t, r in zip(add["team"], add["power_rating"]))
    print(f"  SP+ priors for {len(add)} team(s) new to FBS: {named}")
    return pd.concat([baseline, add[baseline.columns]], ignore_index=True)


def load_priors(season: int, api_key: str = None):
    """Preseason baseline to anchor against, or None to use the old blend."""
    if season < FIRST_ANCHORED_SEASON:
        return None
    path = RATINGS_DIR / f"ratings_{season}_preseason.csv"
    if not path.exists():
        return None
    baseline = pd.read_csv(path)
    if api_key:
        baseline = sp_plus_priors(season, baseline, api_key)
    return baseline


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


def _is_degenerate(row, eps: float = 1e-6) -> bool:
    """True if a team's in-season rating carries no opponent-adjusted signal.

    The iterative adjustment drives every component to exactly league average
    when a team's opponents share no opponents of their own, which is the norm
    in week 1. Such a rating is indistinguishable from "perfectly average" and
    must not be blended.
    """
    components = ("epa_rating_cur", "srs_cur", "adj_off_ppa_cur", "adj_def_ppa_cur")
    seen = False
    for col in components:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        seen = True
        if abs(float(val)) > eps:
            return False
    return seen


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
    skipped_new = []
    for _, row in merged.iterrows():
        games = row.get("games_cur") if pd.notna(row.get("games_cur")) else 0
        games = int(games) if not np.isnan(float(games)) else 0

        has_current = pd.notna(row.get("power_rating_cur"))
        has_preseason = pd.notna(row.get("power_rating_pre"))

        if not has_preseason and not has_current:
            continue

        # Through week 1 every team has exactly one FBS opponent, so the
        # opponent adjustment has nothing to solve against: each team's only
        # opponent is its exact mirror and every adjusted component converges
        # to league average. Blending that in shrinks a team toward zero with
        # no information about whether it won, so hold such teams at their
        # preseason prior until a shared-opponent graph exists (week 2).
        informative = has_current and not _is_degenerate(row)

        if not has_preseason:
            # New to FBS, so there is no prior to fall back on. Without a
            # usable in-season rating there is no basis for rating them at
            # all — leaving them in would plant them at exactly average.
            if not informative:
                skipped_new.append(row.get("team_cur"))
                continue
            w = 1.0
        else:
            w = min(games * per_game_step, 1.0) if informative else 0.0

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
    n_held = int(sum(1 for _, r in merged.iterrows()
                     if pd.notna(r.get("power_rating_cur")) and _is_degenerate(r)))
    print(f"  Blend result: {n_inseason} teams blending in-season, {n_preseason_only} on preseason only")
    if n_held:
        print(f"  ({n_held} played but held at preseason — no shared-opponent signal yet)")
    if skipped_new:
        print(f"  (omitted, new to FBS with no usable rating yet: {', '.join(sorted(skipped_new))})")

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

    # Match generate_weekly.py: teams enter the in-season rating as soon as they
    # have any FBS games. The blend weight (games * 0.15) damps small samples
    # heavily, so early ratings stay anchored to the preseason prior. Leaving
    # this at the default 6 would freeze the season table on preseason until
    # mid-October while the weekly snapshots blended from week 1.
    config.min_fbs_games = 1

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting ratings update")

    try:
        priors = None if args.no_blend else load_priors(args.season, config.api_key)

        # Fetched once and reused: the anchored ratings and the preseason-free
        # companion below are two views of the same games.
        prefetched = (fetch_games(args.season, config),
                      fetch_all_plays(args.season, config))

        ratings = calculate_ratings(
            args.season, config,
            prefetched=prefetched,
            priors=priors,
            prior_weight=PRIOR_WEIGHT if priors is not None else 0.0,
            residual_cap=RESIDUAL_CAP if priors is not None else None,
        )

        if priors is not None:
            ratings = carry_unplayed(ratings, priors)
        elif not args.no_blend:
            # Pre-2026 seasons keep the original blend-after-the-fact path.
            print(f"\n  Blending with preseason baseline...")
            ratings = blend_with_preseason(ratings, args.season)

        # Strength of record and strength of schedule must describe what a
        # team actually did, so they cannot read a rating carrying preseason
        # weight. Write a preseason-free companion for them to consume; the
        # anchored file stays the published/predictive one.
        if priors is not None:
            print("\n  Building preseason-free ratings for SOR/SOS...")
            in_season = calculate_ratings(args.season, config,
                                          prefetched=prefetched)
            # Spread alone is not enough: in week 1 a handful of real ratings
            # among a league of zeros still clears any variance threshold. What
            # matters is whether most teams have signal at all.
            vals = pd.to_numeric(in_season.get("power_rating"), errors="coerce")
            flat = float((vals.abs() < 0.05).mean()) if len(in_season) else 1.0
            companion = RATINGS_DIR / f"ratings_{args.season}_inseason.csv"
            if len(in_season) > 0 and flat < 0.10:
                in_season.to_csv(companion, index=False)
                print(f"  Saved {len(in_season)} teams to {companion.name}")
            else:
                # Early weeks collapse to league average without a prior, which
                # is exactly the degeneracy that motivated anchoring. A file of
                # zeros would be worse than none, so leave it absent and let
                # generate_sor refuse rather than rate a resume off noise.
                if companion.exists():
                    companion.unlink()
                print(f"  Preseason-free ratings still degenerate "
                      f"({flat*100:.0f}% of teams at league average); "
                      f"companion not written")

        # Save to CSV
        output_path = save_ratings(ratings, args.season, config)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise

    print(f"\nAll 1 season(s) updated successfully.")


if __name__ == "__main__":
    main()
