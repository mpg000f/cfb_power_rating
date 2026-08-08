#!/usr/bin/env python3
"""
Generate CFB Strength of Record (SOR) for a season.

SOR answers "how hard was this record to earn?" It drops a benchmark team --
the average top-25 team by that season's own ratings -- into each team's actual
schedule and asks how likely it would be to match or beat that team's record.

Two scopes are written, and they answer different questions:

  full     Whole season including bowls and the CFP, scored against the
           published ratings_{season}.csv. Written to sor_{season}.csv, and
           this is what the website table shows. Its record matches the
           Record column on the site.

  regular  Regular season only (seasonType == 'regular'), which includes
           conference championship week and Army-Navy but excludes bowls and
           the CFP. Ratings are RECOMPUTED from regular season games so no
           postseason result leaks backwards into a September resume. Written
           to sor_{season}_regular.csv. This is the "going into the playoff"
           view and is the slower of the two, since it refetches play-by-play.

Per-game win probability uses a normal model calibrated on 8,536 FBS-vs-FBS
games from 2014-2025 (see CALIBRATION below): Brier 0.1422, log loss 0.4336.

    expected margin = SLOPE * (benchmark - opponent) + home field
    win probability = Phi(expected margin / SIGMA)

Those per-game probabilities feed a Poisson-binomial to get the full
distribution of records the benchmark could post against the schedule. SOR is
where the team's actual win total falls in that distribution, 0-100, higher
meaning fewer benchmark seasons match it.

Usage:
    python generate_sor.py --season 2024              # both scopes
    python generate_sor.py --all --scope full
    python generate_sor.py --season 2024 --scope regular
"""

import argparse
import os
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from power_rating import (RatingConfig, calculate_ratings, calculate_srs,
                          fetch_games, fetch_all_plays, get_fbs_teams)

RATINGS_DIR = Path(__file__).parent / "historical_ratings"
SOR_DIR = RATINGS_DIR / "sor"

# CALIBRATION -- fit on 2014-2025 FBS-vs-FBS regular season games using this
# model's own final ratings. Rerun calibrate_sor.py if the rating scale ever
# changes; SLOPE drifting from 1.0 means ratings under/overstate margins.
SLOPE = 1.107    # margin per point of rating difference
HFA = 2.34       # home field advantage, points
SIGMA = 13.41    # residual std of game margin, points
FCS_RATING = -28.4   # fallback when a season has too few FBS-vs-FCS games
MIN_FCS_GAMES = 20


def load_api_key() -> str:
    key = os.environ.get("CFB_API_KEY", "")
    if not key:
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            cfg = {}
            exec(open(config_path).read(), cfg)
            key = cfg.get("CFB_API_KEY", "")
    return key


def win_probability(margin: float) -> float:
    """Probability of winning given an expected margin in points."""
    return 0.5 * (1 + erf(margin / (SIGMA * sqrt(2))))


def poisson_binomial(probs: list) -> np.ndarray:
    """Distribution over total wins for independent games with these win probs."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for i, p in enumerate(probs):
        # Walk backwards so each game's update reads the pre-update values.
        dist[1:i + 2] = dist[1:i + 2] * (1 - p) + dist[0:i + 1] * p
        dist[0] *= (1 - p)
    return dist


def regular_season_inputs(season: int, config: RatingConfig) -> tuple:
    """Ratings and games through the end of the regular season; returns (ratings, games, srs)."""
    games = fetch_games(season, config)
    plays = fetch_all_plays(season, config)

    reg_games = games[games["seasonType"].astype(str).str.lower() == "regular"].copy()
    if len(plays) == 0 or "season_type" not in plays.columns:
        # Season hasn't been played yet (or has no play-by-play on file).
        print("  No play-by-play available; nothing to rate")
        return pd.DataFrame(), reg_games, pd.DataFrame()
    reg_plays = plays[plays["season_type"] == "regular"].copy()
    print(f"  Regular season only: {len(reg_games)} games, {len(reg_plays)} plays")

    # Ratings must be recomputed here -- the published ratings_{season}.csv
    # includes bowls and the CFP, which would leak backwards into the resume.
    ratings = calculate_ratings(season, config, prefetched=(reg_games, reg_plays))
    # SRS needs only scores, so it covers FBS teams that lack the play-by-play
    # volume to earn a power rating (an issue in 2005-2007).
    srs = calculate_srs(reg_games, get_fbs_teams(reg_games))
    return ratings, reg_games, srs


def full_season_inputs(season: int, config: RatingConfig) -> tuple:
    """Published full-season ratings and every game; returns (ratings, games, srs)."""
    path = RATINGS_DIR / f"ratings_{season}.csv"
    if not path.exists():
        print(f"  No ratings_{season}.csv on disk")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # No recompute needed: ratings_{season}.csv already covers the full season.
    ratings = pd.read_csv(path)
    games = fetch_games(season, config)

    if "games" in ratings.columns and pd.to_numeric(ratings["games"], errors="coerce").max() == 0:
        print("  Ratings file is a preseason baseline, not a played season; skipping")
        return pd.DataFrame(), games, pd.DataFrame()
    if games["homePoints"].notna().sum() == 0:
        print("  No completed games; nothing to rate")
        return pd.DataFrame(), games, pd.DataFrame()

    print(f"  Full season: {len(games)} games including bowls and the CFP")
    srs = calculate_srs(games, get_fbs_teams(games))
    return ratings, games, srs


def unrated_fbs_ratings(ratings: pd.DataFrame, srs: pd.DataFrame) -> dict:
    """Approximate power ratings for FBS teams that SRS covers but the model doesn't."""
    rated = set(ratings["team"].str.lower().str.strip())
    srs = srs.copy()
    srs["key"] = srs["team"].str.lower().str.strip()

    overlap = srs[srs["key"].isin(rated)].merge(
        ratings.assign(key=ratings["team"].str.lower().str.strip())[["key", "power_rating"]],
        on="key")
    missing = srs[~srs["key"].isin(rated)]
    if len(missing) == 0:
        return {}
    if len(overlap) < 10:
        print(f"  Too few teams to map SRS onto the rating scale; {len(missing)} left unrated")
        return {}

    # Put SRS on the power rating scale using the teams that have both.
    design = np.column_stack([overlap["srs"].values, np.ones(len(overlap))])
    slope, intercept = np.linalg.lstsq(design, overlap["power_rating"].values, rcond=None)[0]
    approx = {row["key"]: slope * row["srs"] + intercept for _, row in missing.iterrows()}
    print(f"  Approximated {len(approx)} unrated FBS teams from SRS "
          f"(power = {slope:.2f}*srs {intercept:+.2f}); e.g. "
          + ", ".join(f"{k}={v:.1f}" for k, v in list(approx.items())[:3]))
    return approx


def implied_fcs_rating(games: pd.DataFrame, rating_map: dict) -> float:
    """Estimate an FCS opponent's rating from that season's FBS-vs-FCS results."""
    fcs = games[(games["homeClassification"] == "fbs")
                & (games["awayClassification"] == "fcs")
                & games["homePoints"].notna()].copy()
    fcs["hr"] = fcs["homeTeam"].str.lower().str.strip().map(rating_map)
    fcs = fcs[fcs["hr"].notna()]
    if len(fcs) < MIN_FCS_GAMES:
        print(f"  Only {len(fcs)} FBS-vs-FCS games; using fallback FCS rating {FCS_RATING}")
        return FCS_RATING
    hfa = np.where(fcs["neutralSite"].fillna(False), 0.0, HFA)
    margin = fcs["homePoints"] - fcs["awayPoints"]
    # margin = SLOPE * (hr - r_fcs) + hfa  ->  r_fcs = hr - (margin - hfa) / SLOPE
    implied = (fcs["hr"] - (margin - hfa) / SLOPE).mean()
    print(f"  Implied FCS rating: {implied:.1f} (from {len(fcs)} games)")
    return float(implied)


def build_schedule(games: pd.DataFrame) -> pd.DataFrame:
    """Flatten games into one row per team-game with opponent and site."""
    played = games[games["homePoints"].notna() & games["awayPoints"].notna()].copy()
    played["neutral"] = played["neutralSite"].fillna(False)

    home = pd.DataFrame({
        "team": played["homeTeam"], "opponent": played["awayTeam"],
        "opp_class": played["awayClassification"],
        "won": played["homePoints"] > played["awayPoints"],
        "site": np.where(played["neutral"], "neutral", "home"),
    })
    away = pd.DataFrame({
        "team": played["awayTeam"], "opponent": played["homeTeam"],
        "opp_class": played["homeClassification"],
        "won": played["awayPoints"] > played["homePoints"],
        "site": np.where(played["neutral"], "neutral", "away"),
    })
    return pd.concat([home, away], ignore_index=True)


def calculate_sor(season: int, config: RatingConfig, scope: str = "full",
                  benchmark_n: int = 25) -> pd.DataFrame:
    if scope == "regular":
        ratings, games, srs = regular_season_inputs(season, config)
    else:
        ratings, games, srs = full_season_inputs(season, config)
    if len(ratings) == 0:
        return pd.DataFrame()

    rating_map = dict(zip(ratings["team"].str.lower().str.strip(), ratings["power_rating"]))
    # Opponent-only: these teams get a usable rating but no SOR row of their own,
    # since they never earned a power rating to display alongside it.
    opponent_map = dict(rating_map)
    opponent_map.update(unrated_fbs_ratings(ratings, srs))
    power_rank = dict(zip(ratings["team"].str.lower().str.strip(), ratings["rank"]))
    conf_map = dict(zip(ratings["team"].str.lower().str.strip(),
                        ratings.get("conference", pd.Series(index=ratings.index, dtype=object))))
    # Older ratings_{season}.csv files predate the conference column; the games
    # feed carries it either way.
    for team_col, conf_col in [("homeTeam", "homeConference"), ("awayTeam", "awayConference")]:
        if conf_col not in games.columns:
            continue
        for team, conf in games[[team_col, conf_col]].dropna().drop_duplicates().values:
            key = str(team).lower().strip()
            if pd.isna(conf_map.get(key)):
                conf_map[key] = conf

    benchmark = ratings.nlargest(benchmark_n, "power_rating")["power_rating"].mean()
    fcs_rating = implied_fcs_rating(games, rating_map)
    print(f"  Benchmark (avg top-{benchmark_n}): {benchmark:.1f}")

    schedule = build_schedule(games)
    hfa_by_site = {"home": HFA, "away": -HFA, "neutral": 0.0}

    rows = []
    for team, sched in schedule.groupby("team"):
        key = team.lower().strip()
        if key not in rating_map:
            continue  # non-FBS, or too few games to be rated

        probs, opp_ratings = [], []
        for _, g in sched.iterrows():
            opp_key = str(g["opponent"]).lower().strip()
            if g["opp_class"] == "fbs" and opp_key in opponent_map:
                opp = opponent_map[opp_key]
            else:
                opp = fcs_rating  # FCS and the rare lower-division opponent
            opp_ratings.append(opp)
            margin = SLOPE * (benchmark - opp) + hfa_by_site[g["site"]]
            probs.append(win_probability(margin))

        wins = int(sched["won"].sum())
        losses = len(sched) - wins
        dist = poisson_binomial(probs)
        # Percentile of the actual win total within the benchmark's distribution.
        # Half-credit for ties so an exact match sits at the midpoint.
        sor = dist[:wins].sum() + 0.5 * dist[wins]

        rows.append({
            "team": team,
            "conference": conf_map.get(key),
            "sor": round(sor * 100, 1),
            "wins": wins,
            "losses": losses,
            "record": f"{wins}-{losses}",
            "bench_exp_wins": round(float(np.dot(np.arange(len(dist)), dist)), 2),
            "win_diff": round(wins - float(np.dot(np.arange(len(dist)), dist)), 2),
            "avg_opp_rating": round(float(np.mean(opp_ratings)), 1),
            "power_rating": round(float(rating_map[key]), 1),
            "power_rank": int(power_rank[key]),
            "games": len(sched),
        })

    df = pd.DataFrame(rows).sort_values("sor", ascending=False).reset_index(drop=True)
    df.insert(0, "sor_rank", range(1, len(df) + 1))
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate CFB strength of record")
    parser.add_argument("--season", type=int)
    parser.add_argument("--all", action="store_true", help="Every season with ratings on disk")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--benchmark-n", type=int, default=25,
                        help="Benchmark is the average of the top N teams (default: 25)")
    parser.add_argument("--scope", choices=["full", "regular", "both"], default="both",
                        help="full = whole season incl. bowls/CFP (sor_YYYY.csv, drives the site); "
                             "regular = regular season only (sor_YYYY_regular.csv)")
    args = parser.parse_args()

    config = RatingConfig()
    config.api_key = args.api_key or load_api_key()
    if not config.api_key:
        print("Error: no CFB API key (set CFB_API_KEY or config.py)")
        return

    if args.all:
        seasons = sorted(int(p.stem.split("_")[1]) for p in RATINGS_DIR.glob("ratings_[0-9]*.csv")
                         if "preseason" not in p.stem)
    elif args.season:
        seasons = [args.season]
    else:
        print("Error: pass --season YYYY or --all")
        return

    scopes = ["full", "regular"] if args.scope == "both" else [args.scope]

    SOR_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        for scope in scopes:
            df = calculate_sor(season, config, scope, args.benchmark_n)
            if len(df) == 0:
                print(f"{season} ({scope}): no ratings produced, skipping")
                continue
            suffix = "" if scope == "full" else "_regular"
            path = SOR_DIR / f"sor_{season}{suffix}.csv"
            df.to_csv(path, index=False)
            print(f"\n{season} ({scope}): wrote {len(df)} teams -> {path.name}")
            print(df[["sor_rank", "team", "record", "sor", "bench_exp_wins",
                      "avg_opp_rating", "power_rank"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
