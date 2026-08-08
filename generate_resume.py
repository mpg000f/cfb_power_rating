#!/usr/bin/env python3
"""
Generate CFB Resume Score (RES).

RES estimates where the CFP selection committee would rank a team, using only
inputs this repo already produces. Every input comes out of the regular-season
strength of record files, so RES needs no API calls and no external ratings.

Model, fit on 275 committee top-25 team-seasons (2014-2025, excluding 2020):

    RES = -13.6358 + 2.7065*power_z + 2.4101*sor_z + 1.7929*wins + 0.3435*avg_opp

    power_z   season z-score of power_rating across all rated FBS teams
    sor_z     season z-score of sor across all rated FBS teams
    wins      regular season wins
    avg_opp   average opponent power rating (schedule strength)

The regression target is 26 - committee rank, so RES lands on a "committee
points" scale where roughly 25 is a No. 1 team and 1 is No. 25. Values outside
that band are fine -- they just mean a resume better or worse than the top 25.

Why these four. Power rating and SOR are the core hybrid. `wins` earns its place
because SOR is a percentile and saturates -- a 13-0 and a 12-1 team can both sit
near 99 -- while the committee responds to raw wins more linearly. `avg_opp`
earns its place because SOR compresses record and schedule into one number, and
letting schedule carry its own weight recovers signal. Quality-of-win features
(wins over top-25 teams, sum of beaten teams' ratings) were tested and make the
model WORSE: SOR already carries that information.

Fit quality: in-sample R2 0.809, leave-one-year-out R2 0.799, mean absolute rank
error 2.28 spots. That is a dead heat with the six-variable Boyle Resume Ranking
(0.800 / 2.16), which also needs SP+ as an external input. Charging RES for the
feature search that found it, nested CV puts it at 0.794.

Seasons before 2014 are scored by extrapolation -- the committee did not exist,
so there is no target to fit against. The z-scored inputs keep those scores
season-relative and comparable anyway.

Usage:
    python generate_resume.py --all
    python generate_resume.py --season 2025
    python generate_resume.py --refit          # re-derive coefficients, print diagnostics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATINGS_DIR = Path(__file__).parent / "historical_ratings"
SOR_DIR = RATINGS_DIR / "sor"
RES_DIR = RATINGS_DIR / "resume"
COMMITTEE_CSV = Path(__file__).parent / "data" / "cfp_committee_rankings.csv"

# Fit by --refit on data/cfp_committee_rankings.csv. Order matches FEATURES.
INTERCEPT = -13.6358
COEFS = {
    "power_z": 2.7065,
    "sor_z": 2.4101,
    "wins": 1.7929,
    "avg_opp_rating": 0.3435,
}
FEATURES = list(COEFS)


def load_season(season: int) -> pd.DataFrame:
    """Load a season's regular-season SOR file and add the within-season z-scores."""
    path = SOR_DIR / f"sor_{season}_regular.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Standardise across every rated FBS team that season, so RES means the same
    # thing in a top-heavy year as in a flat one.
    df["power_z"] = (df["power_rating"] - df["power_rating"].mean()) / df["power_rating"].std()
    df["sor_z"] = (df["sor"] - df["sor"].mean()) / df["sor"].std()
    df["season"] = season
    df["key"] = df["team"].str.lower().str.strip()
    return df


def score(df: pd.DataFrame, intercept: float = INTERCEPT, coefs: dict = None) -> pd.Series:
    """Apply the RES formula to a season frame."""
    coefs = coefs or COEFS
    out = pd.Series(intercept, index=df.index, dtype=float)
    for name, weight in coefs.items():
        out += weight * df[name]
    return out


def load_training() -> pd.DataFrame:
    """Join committee rankings to their season's features."""
    committee = pd.read_csv(COMMITTEE_CSV)
    committee["key"] = committee["team"].str.lower().str.strip()

    frames = [load_season(s) for s in sorted(committee["season"].unique())]
    frames = [f for f in frames if len(f)]
    features = pd.concat(frames, ignore_index=True)

    df = committee.merge(features, on=["season", "key"], how="left",
                         suffixes=("", "_sor"))
    unmatched = df[df["power_z"].isna()]
    if len(unmatched):
        print(f"  Warning: {len(unmatched)} committee teams had no SOR row")
        print(unmatched[["season", "team"]].to_string(index=False))
        df = df[df["power_z"].notna()]
    df["target"] = 26 - df["committee_rank"]
    return df


def refit() -> tuple:
    """Re-derive coefficients from the committee data and report fit quality."""
    df = load_training()
    X = np.column_stack([np.ones(len(df))] + [df[f].values for f in FEATURES])
    y = df["target"].values

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    in_r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid) / (len(y) - X.shape[1])))

    # Leave-one-year-out: the only honest read, since seasons are the natural fold.
    oos = np.full(len(y), np.nan)
    for season in df["season"].unique():
        test = (df["season"] == season).values
        b, *_ = np.linalg.lstsq(X[~test], y[~test], rcond=None)
        oos[test] = X[test] @ b
    oos_r2 = 1 - ((y - oos) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    tmp = df.assign(pred=oos)
    tmp["pred_rank"] = tmp.groupby("season")["pred"].rank(ascending=False)
    err = (tmp["pred_rank"] - tmp["committee_rank"]).abs()

    seasons = ", ".join(str(int(s)) for s in sorted(df["season"].unique()))
    print(f"\nFit on {len(df)} committee team-seasons "
          f"({df['season'].nunique()} seasons: {seasons})")
    print(f"  in-sample R2 {in_r2:.4f}   leave-one-year-out R2 {oos_r2:.4f}")
    print(f"  mean absolute rank error {err.mean():.2f} spots, median {err.median():.1f}")
    print("\n  coefficients:")
    for name, b, s in zip(["Intercept"] + FEATURES, beta, se):
        print(f"    {name:<16}{b:>10.4f}   se {s:.4f}   t {b / s:>6.2f}")
    print("\n  per-year mean absolute rank error:")
    for season, g in tmp.groupby("season"):
        e = (g["pred_rank"] - g["committee_rank"]).abs()
        print(f"    {season}: {e.mean():.2f}")
    return float(beta[0]), dict(zip(FEATURES, beta[1:]))


def build_season(season: int, intercept: float, coefs: dict,
                 committee: pd.DataFrame) -> pd.DataFrame:
    df = load_season(season)
    if len(df) == 0:
        return pd.DataFrame()

    df["res"] = score(df, intercept, coefs).round(2)
    df = df.sort_values("res", ascending=False).reset_index(drop=True)
    df["res_rank"] = range(1, len(df) + 1)

    known = committee[committee["season"] == season]
    rank_by_key = dict(zip(known["key"], known["committee_rank"]))
    df["committee_rank"] = df["key"].map(rank_by_key)
    df["vs_committee"] = df["committee_rank"] - df["res_rank"]

    cols = ["res_rank", "team", "conference", "res", "record", "wins", "losses",
            "power_rating", "power_rank", "sor", "sor_rank", "avg_opp_rating",
            "committee_rank", "vs_committee"]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="Generate CFB resume scores")
    parser.add_argument("--season", type=int)
    parser.add_argument("--all", action="store_true", help="Every season with a SOR file")
    parser.add_argument("--refit", action="store_true",
                        help="Re-derive coefficients from committee data and use them")
    args = parser.parse_args()

    intercept, coefs = INTERCEPT, COEFS
    if args.refit:
        intercept, coefs = refit()
        print("\n  (paste these into INTERCEPT/COEFS to make them the default)")

    if args.all:
        seasons = sorted(int(p.stem.split("_")[1])
                         for p in SOR_DIR.glob("sor_[0-9]*_regular.csv"))
    elif args.season:
        seasons = [args.season]
    elif args.refit:
        return  # --refit on its own just reports
    else:
        print("Error: pass --season YYYY, --all, or --refit")
        return

    committee = pd.read_csv(COMMITTEE_CSV)
    committee["key"] = committee["team"].str.lower().str.strip()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        df = build_season(season, intercept, coefs, committee)
        if len(df) == 0:
            print(f"{season}: no SOR file, skipping")
            continue
        path = RES_DIR / f"res_{season}.csv"
        df.to_csv(path, index=False)
        matched = df["committee_rank"].notna().sum()
        note = f", {matched} with committee ranks" if matched else ""
        print(f"\n{season}: wrote {len(df)} teams -> {path.name}{note}")
        print(df[["res_rank", "team", "record", "res", "sor", "power_rank",
                  "committee_rank"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
