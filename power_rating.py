"""
College Football Power Rating System
=====================================
A PPA (EPA) and success rate-based power rating with iterative opponent adjustment.

Features:
- Play-by-play PPA aggregated per team
- Success rate calculated from down/distance thresholds
- Iterative opponent adjustment (15 rounds)
- Recency weighting for in-season updates

Data source: collegefootballdata.com API (play-by-play)

Success Rate Definition:
- 1st down: gain >= 50% of yards to go
- 2nd down: gain >= 70% of yards to go
- 3rd/4th down: convert (gain >= 100% of yards to go)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
import requests
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RatingConfig:
    """Configurable weights and parameters for the CFB power rating system."""

    # Component weights for final rating
    weight_ppa: float = 0.70         # PPA-based efficiency
    weight_success_rate: float = 0.30  # Success rate consistency

    # Opponent adjustment parameters
    opp_adjust_iterations: int = 15  # Number of iterations for convergence

    # Recency weighting
    recency_full_weight_games: int = 4  # Most recent N games at full weight
    recency_weight_min: float = 0.6     # Minimum weight for early season games

    # FBS filtering
    min_fbs_games: int = 6  # Minimum games to be included

    # API configuration
    api_key: str = ""
    api_base_url: str = "https://api.collegefootballdata.com"

    # Paths
    ratings_dir: str = "historical_ratings"


# =============================================================================
# API DATA FETCHING
# =============================================================================

def get_api_headers(config: RatingConfig) -> dict:
    """Get headers for College Football Data API requests."""
    return {
        "Authorization": f"Bearer {config.api_key}",
        "Accept": "application/json"
    }


def fetch_games(season: int, config: RatingConfig) -> pd.DataFrame:
    """Fetch game data for a season."""
    print(f"  Fetching games for {season}...")

    url = f"{config.api_base_url}/games"
    params = {"year": season, "seasonType": "regular", "division": "fbs"}

    response = requests.get(url, headers=get_api_headers(config), params=params)
    response.raise_for_status()
    games = pd.DataFrame(response.json())

    # Also get postseason
    params["seasonType"] = "postseason"
    response = requests.get(url, headers=get_api_headers(config), params=params)
    if response.status_code == 200:
        postseason = pd.DataFrame(response.json())
        if len(postseason) > 0:
            games = pd.concat([games, postseason], ignore_index=True)

    print(f"    Loaded {len(games)} games")
    return games


def fetch_plays_for_week(season: int, week: int, season_type: str, config: RatingConfig) -> pd.DataFrame:
    """Fetch play-by-play data for a specific week."""
    url = f"{config.api_base_url}/plays"
    params = {
        "year": season,
        "week": week,
        "seasonType": season_type,
        "classification": "fbs"
    }

    response = requests.get(url, headers=get_api_headers(config), params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return pd.DataFrame()


def fetch_all_plays(season: int, config: RatingConfig) -> pd.DataFrame:
    """Fetch all play-by-play data for a season."""
    print(f"  Fetching play-by-play data for {season}...")

    all_plays = []

    # Regular season (weeks 1-15)
    for week in range(1, 16):
        time.sleep(0.3)  # Rate limiting
        plays = fetch_plays_for_week(season, week, "regular", config)
        if len(plays) > 0:
            plays["week"] = week
            plays["season_type"] = "regular"
            all_plays.append(plays)
            print(f"    Week {week}: {len(plays)} plays")

    # Postseason (week 1 = bowls)
    time.sleep(0.3)
    plays = fetch_plays_for_week(season, 1, "postseason", config)
    if len(plays) > 0:
        plays["week"] = 16  # Treat as week 16 for recency
        plays["season_type"] = "postseason"
        all_plays.append(plays)
        print(f"    Postseason: {len(plays)} plays")

    if all_plays:
        df = pd.concat(all_plays, ignore_index=True)
        print(f"    Total: {len(df)} plays")
        return df

    return pd.DataFrame()


# =============================================================================
# PLAY PROCESSING
# =============================================================================

def is_successful_play(row) -> bool:
    """
    Determine if a play was successful based on down and distance.

    Success thresholds:
    - 1st down: gain >= 50% of yards to go
    - 2nd down: gain >= 70% of yards to go
    - 3rd/4th down: gain >= 100% of yards to go (conversion)
    """
    down = row.get("down")
    distance = row.get("distance")
    yards_gained = row.get("yardsGained", 0)

    if pd.isna(down) or pd.isna(distance) or distance <= 0:
        return False

    if down == 1:
        return yards_gained >= 0.5 * distance
    elif down == 2:
        return yards_gained >= 0.7 * distance
    elif down in [3, 4]:
        return yards_gained >= distance
    else:
        return False


def process_plays(plays: pd.DataFrame) -> tuple:
    """
    Process play-by-play data to calculate per-game team metrics.

    Returns tuple of (game_stats DataFrame, ppa_std float):
    - game_id, week, team, opponent
    - off_ppa, def_ppa (average PPA per play)
    - off_success_rate, def_success_rate
    - off_plays, def_plays (play counts)
    - ppa_std: standard deviation of raw PPA (for diagnostics)
    """
    if len(plays) == 0:
        return pd.DataFrame(), 1.0

    # Filter to standard downs (exclude kickoffs, punts, etc.)
    # Keep plays with valid down and PPA
    standard_plays = plays[
        plays["down"].notna() &
        plays["ppa"].notna() &
        plays["offense"].notna() &
        plays["defense"].notna()
    ].copy()

    if len(standard_plays) == 0:
        return pd.DataFrame(), 1.0

    # Calculate PPA standard deviation for diagnostics
    ppa_std = standard_plays["ppa"].std()

    # Note: Era differences in PPA scale are handled by the scaling factor
    # in calculate_power_rating (scale=160 for pre-2015, scale=80 for 2015+)

    # Calculate success for each play
    standard_plays["is_success"] = standard_plays.apply(is_successful_play, axis=1)

    # Aggregate by game and team (offense)
    off_stats = standard_plays.groupby(["gameId", "week", "offense", "defense"]).agg(
        off_ppa=("ppa", "mean"),
        off_success_rate=("is_success", "mean"),
        off_plays=("ppa", "count")
    ).reset_index()
    off_stats = off_stats.rename(columns={"offense": "team", "defense": "opponent"})

    # Aggregate by game and team (defense)
    def_stats = standard_plays.groupby(["gameId", "week", "defense", "offense"]).agg(
        def_ppa=("ppa", "mean"),
        def_success_rate=("is_success", "mean"),
        def_plays=("ppa", "count")
    ).reset_index()
    def_stats = def_stats.rename(columns={"defense": "team", "offense": "opponent"})

    # Merge offense and defense stats
    game_stats = off_stats.merge(
        def_stats[["gameId", "team", "def_ppa", "def_success_rate", "def_plays"]],
        on=["gameId", "team"],
        how="outer"
    )

    # Rename gameId for consistency
    game_stats = game_stats.rename(columns={"gameId": "game_id"})

    return game_stats, ppa_std


def get_fbs_teams(games: pd.DataFrame) -> set:
    """Get set of FBS team names from game data (excludes FCS teams)."""
    fbs_teams = set()

    if "homeClassification" in games.columns:
        # Get home teams that are FBS
        fbs_home = games[games["homeClassification"] == "fbs"]["homeTeam"].dropna().unique()
        fbs_teams.update(fbs_home)

        # Get away teams that are FBS
        fbs_away = games[games["awayClassification"] == "fbs"]["awayTeam"].dropna().unique()
        fbs_teams.update(fbs_away)
    else:
        # Fallback if no classification data
        home_teams = set(games["homeTeam"].dropna().unique())
        away_teams = set(games["awayTeam"].dropna().unique())
        fbs_teams = home_teams | away_teams

    return fbs_teams


def filter_fbs_games(game_data: pd.DataFrame, fbs_teams: set) -> pd.DataFrame:
    """Filter to only games between FBS teams."""
    mask = (game_data["team"].isin(fbs_teams)) & (game_data["opponent"].isin(fbs_teams))
    return game_data[mask].copy()


# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

def calculate_recency_weights(game_data: pd.DataFrame, config: RatingConfig) -> pd.DataFrame:
    """
    Add recency weights to game data.
    Most recent N games get full weight, earlier games decay linearly.
    """
    df = game_data.copy()

    if "week" not in df.columns:
        df["recency_weight"] = 1.0
        return df

    weights = []
    for team in df["team"].unique():
        team_games = df[df["team"] == team].sort_values("week")
        n_games = len(team_games)

        for i, (idx, _) in enumerate(team_games.iterrows()):
            games_ago = n_games - 1 - i

            if games_ago < config.recency_full_weight_games:
                weight = 1.0
            else:
                decay_games = games_ago - config.recency_full_weight_games
                max_decay_games = n_games - config.recency_full_weight_games
                if max_decay_games > 0:
                    weight = config.recency_weight_min + (1.0 - config.recency_weight_min) * (1 - decay_games / max_decay_games)
                else:
                    weight = 1.0

            weights.append({"idx": idx, "recency_weight": weight})

    weight_df = pd.DataFrame(weights).set_index("idx")
    df["recency_weight"] = df.index.map(weight_df["recency_weight"])

    return df


# =============================================================================
# OPPONENT ADJUSTMENT (ITERATIVE)
# =============================================================================

def calculate_opponent_adjusted_metrics(
    game_data: pd.DataFrame,
    config: RatingConfig
) -> pd.DataFrame:
    """
    Calculate opponent-adjusted PPA and success rate.

    Uses iterative algorithm:
    1. Start with raw averages for each team
    2. For each game, adjust based on opponent strength
    3. Iterate until convergence

    Adjustment (additive):
    Adjusted Off PPA = Raw Off PPA + (Avg Def PPA - Opp Adj Def PPA)
    Adjusted Def PPA = Raw Def PPA - (Opp Adj Off PPA - Avg Off PPA)
    """
    df = game_data.copy()
    df = calculate_recency_weights(df, config)

    teams = df["team"].unique()

    def weighted_mean(group, col):
        weights = group["recency_weight"].fillna(1.0)
        values = group[col]
        valid = ~values.isna() & ~np.isinf(values)
        if valid.sum() > 0 and weights[valid].sum() > 0:
            return np.average(values[valid], weights=weights[valid])
        return 0.0

    # Initialize with raw weighted averages
    team_stats = []
    for team in teams:
        team_games = df[df["team"] == team]
        # Total plays for shrinkage calculation
        total_plays = team_games["off_plays"].sum() + team_games["def_plays"].sum()
        team_stats.append({
            "team": team,
            "games": len(team_games),
            "total_plays": total_plays,
            "raw_off_ppa": weighted_mean(team_games, "off_ppa"),
            "raw_def_ppa": weighted_mean(team_games, "def_ppa"),
            "raw_off_sr": weighted_mean(team_games, "off_success_rate"),
            "raw_def_sr": weighted_mean(team_games, "def_success_rate"),
        })

    team_stats = pd.DataFrame(team_stats)

    # Baselines (0 for PPA, league average for SR)
    avg_off_ppa = 0.0
    avg_def_ppa = 0.0
    avg_off_sr = team_stats["raw_off_sr"].mean()
    avg_def_sr = team_stats["raw_def_sr"].mean()

    # Initialize adjusted values
    team_stats["adj_off_ppa"] = team_stats["raw_off_ppa"]
    team_stats["adj_def_ppa"] = team_stats["raw_def_ppa"]
    team_stats["adj_off_sr"] = team_stats["raw_off_sr"]
    team_stats["adj_def_sr"] = team_stats["raw_def_sr"]

    # Adjustment strength: 1.0 for stable SRS-style convergence
    adj_strength = 1.0

    print(f"  Running opponent adjustment ({config.opp_adjust_iterations} iterations, strength={adj_strength})...")

    for iteration in range(config.opp_adjust_iterations):
        prev_off_ppa = team_stats["adj_off_ppa"].copy()

        # Create lookups
        off_ppa_lookup = dict(zip(team_stats["team"], team_stats["adj_off_ppa"]))
        def_ppa_lookup = dict(zip(team_stats["team"], team_stats["adj_def_ppa"]))
        off_sr_lookup = dict(zip(team_stats["team"], team_stats["adj_off_sr"]))
        def_sr_lookup = dict(zip(team_stats["team"], team_stats["adj_def_sr"]))

        new_adj = {"off_ppa": [], "def_ppa": [], "off_sr": [], "def_sr": []}

        for team in team_stats["team"]:
            team_games = df[df["team"] == team]

            if len(team_games) == 0:
                for key in new_adj:
                    new_adj[key].append(0.0)
                continue

            adj_values = {"off_ppa": [], "def_ppa": [], "off_sr": [], "def_sr": []}
            weights = []

            for _, game in team_games.iterrows():
                opp = game["opponent"]
                w = game.get("recency_weight", 1.0)

                opp_adj_def_ppa = def_ppa_lookup.get(opp, avg_def_ppa)
                opp_adj_off_ppa = off_ppa_lookup.get(opp, avg_off_ppa)
                opp_adj_def_sr = def_sr_lookup.get(opp, avg_def_sr)
                opp_adj_off_sr = off_sr_lookup.get(opp, avg_off_sr)

                # Adjust PPA (additive with strength multiplier)
                if not pd.isna(game.get("off_ppa")):
                    adj_off = game["off_ppa"] + adj_strength * (avg_def_ppa - opp_adj_def_ppa)
                    adj_values["off_ppa"].append(adj_off)

                if not pd.isna(game.get("def_ppa")):
                    adj_def = game["def_ppa"] - adj_strength * (opp_adj_off_ppa - avg_off_ppa)
                    adj_values["def_ppa"].append(adj_def)

                # Adjust success rate (with strength multiplier)
                if not pd.isna(game.get("off_success_rate")):
                    adj_off_sr = game["off_success_rate"] + adj_strength * (avg_def_sr - opp_adj_def_sr)
                    adj_values["off_sr"].append(adj_off_sr)

                if not pd.isna(game.get("def_success_rate")):
                    adj_def_sr = game["def_success_rate"] - adj_strength * (opp_adj_off_sr - avg_off_sr)
                    adj_values["def_sr"].append(adj_def_sr)

                weights.append(w)

            for key in new_adj:
                vals = adj_values[key]
                if vals and len(weights[:len(vals)]) > 0:
                    new_adj[key].append(np.average(vals, weights=weights[:len(vals)]))
                else:
                    new_adj[key].append(0.0)

        team_stats["adj_off_ppa"] = new_adj["off_ppa"]
        team_stats["adj_def_ppa"] = new_adj["def_ppa"]
        team_stats["adj_off_sr"] = new_adj["off_sr"]
        team_stats["adj_def_sr"] = new_adj["def_sr"]

        # Normalize PPA to mean 0 (required each iteration for convergence)
        team_stats["adj_off_ppa"] = team_stats["adj_off_ppa"] - team_stats["adj_off_ppa"].mean()
        team_stats["adj_def_ppa"] = team_stats["adj_def_ppa"] - team_stats["adj_def_ppa"].mean()

        # NOTE: Success rate centering removed from loop to avoid double-centering
        # Center once after loop completes (see below)

        # Check convergence
        ppa_change = np.abs(team_stats["adj_off_ppa"] - prev_off_ppa).mean()

        if iteration % 3 == 0 or iteration == config.opp_adjust_iterations - 1:
            print(f"    Iteration {iteration + 1}: avg change = {ppa_change:.6f}")

        if ppa_change < 0.0001:
            print(f"    Converged at iteration {iteration + 1}")
            break

    # Center success rates once after opponent adjustment completes (avoid double-centering)
    team_stats["adj_off_sr"] = team_stats["adj_off_sr"] - team_stats["adj_off_sr"].mean() + avg_off_sr
    team_stats["adj_def_sr"] = team_stats["adj_def_sr"] - team_stats["adj_def_sr"].mean() + avg_def_sr

    # Apply shrinkage/regression to prior based on sample size
    # This handles: early-season noise, low-sample teams, blowout volatility
    min_plays_full_weight = 800  # ~60 plays/game * 13 games = full season
    preseason_prior = 0.0  # Average team (could use recruiting rankings)

    shrinkage_weight = np.minimum(team_stats["total_plays"] / min_plays_full_weight, 1.0)

    team_stats["adj_off_ppa"] = (
        shrinkage_weight * team_stats["adj_off_ppa"] +
        (1 - shrinkage_weight) * preseason_prior
    )
    team_stats["adj_def_ppa"] = (
        shrinkage_weight * team_stats["adj_def_ppa"] +
        (1 - shrinkage_weight) * preseason_prior
    )
    team_stats["adj_off_sr"] = (
        shrinkage_weight * team_stats["adj_off_sr"] +
        (1 - shrinkage_weight) * avg_off_sr
    )
    team_stats["adj_def_sr"] = (
        shrinkage_weight * team_stats["adj_def_sr"] +
        (1 - shrinkage_weight) * avg_def_sr
    )

    return team_stats


# =============================================================================
# POWER RATING CALCULATION
# =============================================================================

def calculate_power_rating(team_stats: pd.DataFrame, config: RatingConfig,
                          baseline_points: float = 28.0, game_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate final power rating (IPR) from adjusted metrics.

    IPR = Off Rating - Def Rating = expected margin vs average team

    Off Rating = expected points scored vs average defense
    Def Rating = expected points allowed vs average offense

    Components weighted per config (default 70% PPA, 30% Success Rate)
    Dynamic scaling based on game-level EPA variance to achieve consistent spread

    Args:
        baseline_points: Average points per team (calculated from games data)
        game_data: Game-level data for dynamic scaling calculation
    """
    df = team_stats.copy()

    # Calculate scale from game-level net EPA variance
    # Using game-level data (not team-level means which compress after opponent adjustment)
    # Target: best teams ~+30-35, worst ~-20-25, total spread ~50-70 points
    if game_data is not None and len(game_data) > 0:
        game_net_epa = game_data["off_ppa"] - game_data["def_ppa"]
        epa_std = game_net_epa.std()
        scale = 25.0 / epa_std if epa_std > 0 else 100.0
    else:
        scale = 100.0  # Fallback for missing data

    ppa_weight = config.weight_ppa
    sr_weight = config.weight_success_rate
    assert abs(ppa_weight + sr_weight - 1.0) < 1e-6, "Weights must sum to 1.0"

    # Get league average success rates for deviation calculation
    avg_off_sr = df["adj_off_sr"].mean()
    avg_def_sr = df["adj_def_sr"].mean()

    # Calculate offensive component (points above/below average)
    df["off_ppa_contrib"] = df["adj_off_ppa"] * scale * ppa_weight
    df["off_sr_contrib"] = (df["adj_off_sr"] - avg_off_sr) * scale * sr_weight
    df["off_adjustment"] = df["off_ppa_contrib"] + df["off_sr_contrib"]

    # Calculate defensive component (points above/below average ALLOWED)
    df["def_ppa_contrib"] = df["adj_def_ppa"] * scale * ppa_weight
    df["def_sr_contrib"] = (df["adj_def_sr"] - avg_def_sr) * scale * sr_weight
    df["def_adjustment"] = df["def_ppa_contrib"] + df["def_sr_contrib"]

    # Center the adjustments so average team = baseline
    df["off_adjustment"] = df["off_adjustment"] - df["off_adjustment"].mean()
    df["def_adjustment"] = df["def_adjustment"] - df["def_adjustment"].mean()

    # Off Rating = baseline + adjustment
    df["off_rating"] = baseline_points + df["off_adjustment"]

    # Def Rating = baseline + adjustment
    df["def_rating"] = baseline_points + df["def_adjustment"]

    # IPR = Off Rating - Def Rating = expected margin
    df["power_rating"] = df["off_rating"] - df["def_rating"]

    # Rank teams
    df = df.sort_values("power_rating", ascending=False)
    df["rank"] = range(1, len(df) + 1)

    return df


# =============================================================================
# MAIN RATING PIPELINE
# =============================================================================

def calculate_ratings(season: int, config: RatingConfig) -> pd.DataFrame:
    """
    Main function to calculate CFB power ratings for a season.
    """
    print(f"\n{'='*60}")
    print(f"Calculating CFB Power Ratings for {season}")
    print(f"{'='*60}")

    # Fetch games to get FBS teams and calculate baseline scoring
    games = fetch_games(season, config)
    fbs_teams = get_fbs_teams(games)
    print(f"  Found {len(fbs_teams)} FBS teams")

    # Calculate dynamic baseline from actual FBS scoring
    baseline_points = 28.0  # Default fallback
    if "homeClassification" in games.columns:
        fbs_games = games[
            (games["homeClassification"] == "fbs") &
            (games["awayClassification"] == "fbs")
        ]
        if len(fbs_games) > 0 and "homePoints" in fbs_games.columns:
            avg_home = fbs_games["homePoints"].dropna().mean()
            avg_away = fbs_games["awayPoints"].dropna().mean()
            baseline_points = (avg_home + avg_away) / 2
    print(f"  Season avg scoring: {baseline_points:.1f} ppg")

    # Fetch play-by-play data
    plays = fetch_all_plays(season, config)

    if len(plays) == 0:
        print("  Error: No play data found")
        return pd.DataFrame()

    # Process plays to game-level stats
    game_data, ppa_std = process_plays(plays)
    print(f"  Processed {len(game_data)} team-game records (raw PPA std: {ppa_std:.3f})")

    # Filter to FBS vs FBS
    game_data = filter_fbs_games(game_data, fbs_teams)
    print(f"  FBS vs FBS: {len(game_data)} records")

    # Calculate opponent-adjusted metrics
    team_stats = calculate_opponent_adjusted_metrics(game_data, config)

    # Filter to teams with enough games
    team_stats = team_stats[team_stats["games"] >= config.min_fbs_games]
    print(f"  {len(team_stats)} teams with {config.min_fbs_games}+ games")

    # Calculate power ratings with dynamic baseline and variance-based scaling
    ratings = calculate_power_rating(team_stats, config, baseline_points, game_data)

    # Prepare output
    output_cols = [
        "rank", "team", "power_rating", "off_rating", "def_rating",
        "adj_off_ppa", "adj_def_ppa",
        "adj_off_sr", "adj_def_sr",
        "games"
    ]

    result = ratings[output_cols].copy()
    result = result.round({
        "power_rating": 1,
        "off_rating": 1,
        "def_rating": 1,
        "adj_off_ppa": 3,
        "adj_def_ppa": 3,
        "adj_off_sr": 3,
        "adj_def_sr": 3,
    })

    print(f"\nTop 10 Teams:")
    print(result.head(10).to_string(index=False))

    return result


def save_ratings(ratings: pd.DataFrame, season: int, config: RatingConfig) -> Path:
    """Save ratings to CSV file."""
    ratings_dir = Path(config.ratings_dir)
    ratings_dir.mkdir(exist_ok=True)

    output_path = ratings_dir / f"ratings_{season}.csv"
    ratings.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return output_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Calculate CFB Power Ratings")
    parser.add_argument("--season", type=int, required=True, help="Season year")
    parser.add_argument("--api-key", type=str, help="CFBD API key")

    args = parser.parse_args()

    config = RatingConfig()

    if args.api_key:
        config.api_key = args.api_key
    else:
        config.api_key = os.environ.get("CFB_API_KEY", "")
        if not config.api_key:
            config_path = Path(__file__).parent / "config.py"
            if config_path.exists():
                exec(open(config_path).read())
                config.api_key = globals().get("CFB_API_KEY", "")

    if not config.api_key:
        print("Error: API key required")
        exit(1)

    ratings = calculate_ratings(args.season, config)
    save_ratings(ratings, args.season, config)
