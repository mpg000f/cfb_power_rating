#!/usr/bin/env python3
"""
Fit the SOR win-probability constants (SLOPE, HFA, SIGMA) used by generate_sor.py.

Regresses actual game margin on rating difference across FBS-vs-FBS games, then
reports how well the resulting normal model predicts wins. Rerun this whenever
the rating scale changes and copy the constants into generate_sor.py.

SLOPE is the margin produced per point of rating difference. It should sit near
1.0 -- a value above 1.0 means the ratings understate real margins, and skipping
it makes win probabilities under-confident at both tails.

Usage:
    python calibrate_sor.py
    python calibrate_sor.py --start 2014 --end 2025
"""

import argparse
import os
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RATINGS_DIR = Path(__file__).parent / "historical_ratings"


def load_api_key() -> str:
    key = os.environ.get("CFB_API_KEY", "")
    if not key:
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            cfg = {}
            exec(open(config_path).read(), cfg)
            key = cfg.get("CFB_API_KEY", "")
    return key


def load_games(seasons, api_key: str) -> pd.DataFrame:
    """Fetch regular season games joined to each season's final ratings."""
    headers = {"Authorization": f"Bearer {api_key}"}
    frames = []
    for season in seasons:
        path = RATINGS_DIR / f"ratings_{season}.csv"
        if not path.exists():
            continue
        resp = requests.get("https://api.collegefootballdata.com/games", headers=headers,
                            params={"year": season, "seasonType": "regular"})
        resp.raise_for_status()
        games = pd.DataFrame(resp.json())
        games = games[games["homePoints"].notna() & games["awayPoints"].notna()].copy()

        ratings = pd.read_csv(path)
        rating_map = dict(zip(ratings["team"].str.lower().str.strip(), ratings["power_rating"]))
        games["hr"] = games["homeTeam"].str.lower().str.strip().map(rating_map)
        games["ar"] = games["awayTeam"].str.lower().str.strip().map(rating_map)
        games["season"] = season
        games["margin"] = games["homePoints"] - games["awayPoints"]
        games["neutral"] = games["neutralSite"].fillna(False)
        frames.append(games)
        print(f"  {season}: {len(games)} games")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Calibrate SOR win probability constants")
    parser.add_argument("--start", type=int, default=2014)
    parser.add_argument("--end", type=int, default=2025)
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print("Error: no CFB API key (set CFB_API_KEY or config.py)")
        return

    print(f"Loading {args.start}-{args.end}...")
    data = load_games(range(args.start, args.end + 1), api_key)

    fbs = data[(data["homeClassification"] == "fbs") & (data["awayClassification"] == "fbs")
               & data["hr"].notna() & data["ar"].notna()].copy()
    fbs["diff"] = fbs["hr"] - fbs["ar"]

    # Fit slope and home field jointly on non-neutral games.
    non_neutral = fbs[~fbs["neutral"]]
    design = np.column_stack([non_neutral["diff"].values, np.ones(len(non_neutral))])
    slope, hfa = np.linalg.lstsq(design, non_neutral["margin"].values, rcond=None)[0]

    fbs["pred"] = slope * fbs["diff"] + np.where(fbs["neutral"], 0.0, hfa)
    sigma = (fbs["margin"] - fbs["pred"]).std()

    print(f"\nSLOPE = {slope:.3f}")
    print(f"HFA   = {hfa:.2f}")
    print(f"SIGMA = {sigma:.2f}")
    print(f"  fit on {len(non_neutral)} non-neutral games, {len(fbs)} total")

    prob = fbs["pred"].apply(lambda m: 0.5 * (1 + erf(m / (sigma * sqrt(2)))))
    won = (fbs["margin"] > 0).astype(int)
    brier = ((prob - won) ** 2).mean()
    clipped = prob.clip(1e-9, 1 - 1e-9)
    logloss = -(won * np.log(clipped) + (1 - won) * np.log(1 - clipped)).mean()
    print(f"\nBrier {brier:.4f}   log loss {logloss:.4f}")

    print("\npredicted vs actual win rate:")
    buckets = pd.cut(prob, np.arange(0, 1.01, 0.1))
    table = pd.DataFrame({"n": won.groupby(buckets, observed=True).size(),
                          "pred": prob.groupby(buckets, observed=True).mean(),
                          "actual": won.groupby(buckets, observed=True).mean()})
    for label, row in table.iterrows():
        print(f"  {str(label):14s} n={int(row['n']):5d}  pred {row['pred']:.3f}  "
              f"actual {row['actual']:.3f}  diff {row['actual'] - row['pred']:+.3f}")

    # Implied FCS rating, used as the opponent rating for non-FBS games.
    fcs = data[(data["homeClassification"] == "fbs") & (data["awayClassification"] == "fcs")
               & data["hr"].notna()].copy()
    fcs_hfa = np.where(fcs["neutral"], 0.0, hfa)
    implied = fcs["hr"] - (fcs["margin"] - fcs_hfa) / slope
    print(f"\nFCS_RATING = {implied.mean():.1f}   (from {len(fcs)} FBS-vs-FCS games)")
    print("  by season:")
    for season, value in implied.groupby(fcs["season"]).mean().items():
        print(f"    {season}: {value:.1f}")


if __name__ == "__main__":
    main()
