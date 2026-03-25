#!/usr/bin/env python3
"""
Generate CFB Preseason Power Ratings

Methodology:
  1. Blend 3 prior seasons' ratings (60% most recent, 30% two years ago, 10% three years ago)
  2. Adjust offensive and defensive ratings separately using returning production
  3. Recompute power_rating = adj_off - adj_def

Returning production effect: 1 std dev in RP ≈ 3 points of off/def rating shift.

Usage:
    python generate_preseason.py --target 2026
    python generate_preseason.py --target 2026 --rp-weight 3.0
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

RATINGS_DIR = Path(__file__).parent / "historical_ratings"
RP_DIR = (Path(__file__).parent.parent /
          "College Football Power Ratings" / "Returning Production Data")

# Fallback if the relative path doesn't resolve (e.g., different working dir)
RP_DIR_ALT = (Path.home() / "Documents" /
              "College Football Power Ratings" / "Returning Production Data")


def load_season_ratings(year: int) -> pd.DataFrame | None:
    path = RATINGS_DIR / f"ratings_{year}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Normalize team name to lowercase for merging with RP data
    df["team_lower"] = df["team"].str.lower().str.strip()
    return df


def load_returning_production(year: int) -> pd.DataFrame | None:
    for rp_dir in [RP_DIR, RP_DIR_ALT]:
        path = rp_dir / f"returning_production_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["team_lower"] = df["team"].str.lower().str.strip()
            return df
    return None


def blend_ratings(target_year: int,
                  weights: tuple = (0.6, 0.3, 0.1)) -> pd.DataFrame:
    """
    Blend 3 prior seasons into a single preseason baseline.
    Uses available years gracefully (renormalizes weights if a year is missing).
    """
    years = [target_year - 1, target_year - 2, target_year - 3]
    seasons = []
    actual_weights = []

    for yr, w in zip(years, weights):
        df = load_season_ratings(yr)
        if df is not None:
            seasons.append((yr, df))
            actual_weights.append(w)
        else:
            print(f"  Warning: no ratings found for {yr}, skipping")

    if not seasons:
        raise ValueError(f"No historical ratings found for {target_year - 1} through {target_year - 3}")

    # Renormalize weights
    total_w = sum(actual_weights)
    actual_weights = [w / total_w for w in actual_weights]

    # Collect all teams across all years
    all_teams = pd.concat([df[["team", "team_lower"]] for _, df in seasons]).drop_duplicates("team_lower")

    blended = all_teams.copy()

    for col in ["power_rating", "off_rating", "def_rating", "srs", "epa_rating"]:
        blended[col] = 0.0
        blended[f"{col}_weight_sum"] = 0.0

    for (yr, df), w in zip(seasons, actual_weights):
        merged = blended.merge(
            df[["team_lower", "power_rating", "off_rating", "def_rating", "srs", "epa_rating"]],
            on="team_lower", how="left", suffixes=("", f"_{yr}")
        )
        for col in ["power_rating", "off_rating", "def_rating", "srs", "epa_rating"]:
            col_yr = f"{col}_{yr}"
            has_data = merged[col_yr].notna()
            blended.loc[has_data, col] += merged.loc[has_data, col_yr] * w
            blended.loc[has_data, f"{col}_weight_sum"] += w

    # Re-normalize by actual weight sum (handles teams not present in all years)
    for col in ["power_rating", "off_rating", "def_rating", "srs", "epa_rating"]:
        wsum = blended[f"{col}_weight_sum"]
        blended[col] = blended[col] / wsum.where(wsum > 0, 1.0)
        blended.drop(columns=[f"{col}_weight_sum"], inplace=True)

    # Drop teams with no ratings at all
    blended = blended[blended["power_rating"] != 0].copy()

    print(f"  Blended ratings for {len(blended)} teams using years {[yr for yr, _ in seasons]}")
    print(f"  Weights: {[f'{w:.2f}' for w in actual_weights]}")
    return blended


def apply_returning_production(blended: pd.DataFrame,
                               target_year: int,
                               rp_weight: float = 3.0) -> pd.DataFrame:
    """
    Adjust blended ratings using returning production data.

    rp_weight: points of off/def rating shift per 1 std dev in returning production.
    """
    rp = load_returning_production(target_year)
    if rp is None:
        print(f"  Warning: no returning production data for {target_year}, skipping RP adjustment")
        blended["adj_off"] = blended["off_rating"]
        blended["adj_def"] = blended["def_rating"]
        blended["adj_epa"] = blended["epa_rating"]
        blended["adj_power"] = blended["power_rating"]
        blended["Ret_Prod"] = None
        return blended

    print(f"  Loaded returning production for {len(rp)} teams")

    # Z-score the RP values
    rp["scaled_off"] = (rp["Off_Prod"] - rp["Off_Prod"].mean()) / rp["Off_Prod"].std()
    rp["scaled_def"] = (rp["Def_Prod"] - rp["Def_Prod"].mean()) / rp["Def_Prod"].std()

    df = blended.merge(
        rp[["team_lower", "scaled_off", "scaled_def", "Ret_Prod"]],
        on="team_lower", how="left"
    )

    # Fill missing RP with neutral (0 = no adjustment)
    df["scaled_off"] = df["scaled_off"].fillna(0.0)
    df["scaled_def"] = df["scaled_def"].fillna(0.0)
    df["Ret_Prod"] = df["Ret_Prod"].fillna(rp["Ret_Prod"].mean())

    # Apply adjustments
    # Off: more returning offense → higher off_rating
    df["adj_off"] = df["off_rating"] + rp_weight * df["scaled_off"]
    # Def: more returning defense → lower def_rating (lower = better defense in this system)
    df["adj_def"] = df["def_rating"] - rp_weight * df["scaled_def"]

    # Recompute epa_rating and power_rating from adjusted off/def
    df["adj_epa"] = df["adj_off"] - df["adj_def"]

    # Blend adjusted EPA with blended SRS (60/40 as in the in-season model)
    df["adj_power"] = 0.6 * df["adj_epa"] + 0.4 * df["srs"]

    return df


def build_preseason_ratings(target_year: int,
                             blend_weights: tuple = (0.6, 0.3, 0.1),
                             rp_weight: float = 3.0) -> pd.DataFrame:
    print(f"\nGenerating preseason {target_year} CFB ratings...")

    # Step 1: Blend prior years
    blended = blend_ratings(target_year, blend_weights)

    # Step 2: Adjust for returning production
    df = apply_returning_production(blended, target_year, rp_weight)

    # Step 3: Build output in the same format as in-season ratings
    df = df.sort_values("adj_power", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    output = pd.DataFrame({
        "rank": df["rank"],
        "team": df["team"],
        "power_rating": df["adj_power"].round(1),
        "record": "Preseason",
        "wins": 0,
        "losses": 0,
        "off_rating": df["adj_off"].round(1),
        "def_rating": df["adj_def"].round(1),
        "adj_off_ppa": None,
        "adj_def_ppa": None,
        "adj_off_sr": None,
        "adj_def_sr": None,
        "srs": df["srs"].round(1),
        "epa_rating": df["adj_epa"].round(1),
        "games": 0,
    })

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate CFB preseason power ratings")
    parser.add_argument("--target", type=int, default=2026,
                        help="Target season year (e.g. 2026)")
    parser.add_argument("--rp-weight", type=float, default=3.0,
                        help="Points of off/def rating shift per 1 std dev of returning production (default: 3.0)")
    parser.add_argument("--weights", type=str, default="0.6,0.3,0.1",
                        help="Blend weights for years t-1,t-2,t-3 (comma-separated, default: 0.6,0.3,0.1)")
    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 3:
        print("Error: --weights must have exactly 3 values")
        return

    ratings = build_preseason_ratings(args.target, weights, args.rp_weight)

    # Save the current-display ratings (will be overwritten by blended in-season ratings once season starts)
    output_path = RATINGS_DIR / f"ratings_{args.target}.csv"
    ratings.to_csv(output_path, index=False)

    # Save the permanent preseason baseline (never overwritten — used for in-season blending)
    baseline_path = RATINGS_DIR / f"ratings_{args.target}_preseason.csv"
    ratings.to_csv(baseline_path, index=False)

    print(f"\nSaved preseason {args.target} ratings to: {output_path}")
    print(f"Saved preseason baseline to: {baseline_path}")
    print(f"\nTop 10:")
    print(ratings[["rank", "team", "power_rating", "off_rating", "def_rating"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
