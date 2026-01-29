"""
CFB Power Rating Validation - Smoke Tests
==========================================
Alerting thresholds to catch obviously wrong rankings.

IMPORTANT: These are smoke tests, NOT constraints.
- If a check fails, investigate manually
- Do NOT tune the model to pass validation (that's label leakage)
- These detect symptoms, not root causes
"""

import pandas as pd
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    message: str
    severity: str  # "warning" or "error"


def validate_rating_spread(ratings: pd.DataFrame, min_spread: float = 40.0,
                          max_spread: float = 80.0) -> ValidationResult:
    """
    Check that rating spread is in a sane range.

    Typical CFB spread: 50-70 points between best and worst FBS teams.
    Too narrow = model not differentiating teams.
    Too wide = model is unstable/overfitting.
    """
    if "power_rating" not in ratings.columns:
        return ValidationResult(
            check_name="rating_spread",
            passed=False,
            message="Missing power_rating column",
            severity="error"
        )

    spread = ratings["power_rating"].max() - ratings["power_rating"].min()
    passed = min_spread <= spread <= max_spread

    return ValidationResult(
        check_name="rating_spread",
        passed=passed,
        message=f"Rating spread: {spread:.1f} (expected {min_spread}-{max_spread})",
        severity="warning" if not passed else "info"
    )


def validate_elite_teams_not_buried(ratings: pd.DataFrame, season: int,
                                    elite_teams: Optional[dict] = None) -> ValidationResult:
    """
    Check that historically elite teams for a season aren't ranked too low.

    This is a sanity check - if a national champion is ranked #50, something is wrong.
    """
    # Known elite teams by season (champions, playoff teams)
    default_elite = {
        2019: ["LSU", "Clemson", "Ohio State", "Oklahoma"],
        2020: ["Alabama", "Clemson", "Ohio State", "Notre Dame"],
        2021: ["Georgia", "Alabama", "Michigan", "Cincinnati"],
        2022: ["Georgia", "TCU", "Michigan", "Ohio State"],
        2023: ["Michigan", "Washington", "Alabama", "Florida State"],
        2024: ["Oregon", "Texas", "Penn State", "Notre Dame"],
    }

    if elite_teams is None:
        elite_teams = default_elite

    if season not in elite_teams:
        return ValidationResult(
            check_name="elite_teams",
            passed=True,
            message=f"No elite teams defined for {season}, skipping check",
            severity="info"
        )

    expected_elite = elite_teams[season]
    buried_teams = []

    for team in expected_elite:
        team_row = ratings[ratings["team"] == team]
        if len(team_row) > 0:
            rank = team_row["rank"].iloc[0]
            if rank > 20:  # Elite teams should be top 20 at minimum
                buried_teams.append(f"{team} (#{rank})")

    passed = len(buried_teams) == 0

    if buried_teams:
        message = f"Elite teams ranked too low: {', '.join(buried_teams)}"
    else:
        message = f"All {len(expected_elite)} elite teams in top 20"

    return ValidationResult(
        check_name="elite_teams",
        passed=passed,
        message=message,
        severity="warning" if not passed else "info"
    )


def validate_awful_teams_not_elite(ratings: pd.DataFrame, season: int,
                                   awful_teams: Optional[dict] = None) -> ValidationResult:
    """
    Check that historically bad teams aren't ranked in the top 25.

    This catches cases where the model might be inverting or wildly miscalibrated.
    """
    # Teams that were consistently bad in recent years
    default_awful = {
        2019: ["UMass", "Akron", "New Mexico State", "UTEP"],
        2020: ["UMass", "Kansas", "Vanderbilt", "South Carolina"],
        2021: ["UMass", "UConn", "Arizona", "Vanderbilt"],
        2022: ["UMass", "Colorado", "Arizona", "Northwestern"],
        2023: ["Kent State", "UMass", "Colorado State", "Akron"],
        2024: ["Kent State", "UMass", "Florida Atlantic", "Utah State"],
    }

    if awful_teams is None:
        awful_teams = default_awful

    if season not in awful_teams:
        return ValidationResult(
            check_name="awful_teams",
            passed=True,
            message=f"No awful teams defined for {season}, skipping check",
            severity="info"
        )

    expected_awful = awful_teams[season]
    misranked_teams = []

    for team in expected_awful:
        team_row = ratings[ratings["team"] == team]
        if len(team_row) > 0:
            rank = team_row["rank"].iloc[0]
            if rank <= 25:  # Awful teams should NOT be top 25
                misranked_teams.append(f"{team} (#{rank})")

    passed = len(misranked_teams) == 0

    if misranked_teams:
        message = f"Bad teams ranked too high: {', '.join(misranked_teams)}"
    else:
        message = f"No awful teams in top 25"

    return ValidationResult(
        check_name="awful_teams",
        passed=passed,
        message=message,
        severity="warning" if not passed else "info"
    )


def validate_ratings(ratings: pd.DataFrame, season: int) -> List[ValidationResult]:
    """
    Run all validation checks on ratings.

    Returns list of ValidationResult objects.
    """
    results = []

    results.append(validate_rating_spread(ratings))
    results.append(validate_elite_teams_not_buried(ratings, season))
    results.append(validate_awful_teams_not_elite(ratings, season))

    return results


def print_validation_report(results: List[ValidationResult], season: int) -> bool:
    """
    Print validation report and return True if all checks passed.
    """
    print(f"\n{'='*60}")
    print(f"Validation Report - {season}")
    print(f"{'='*60}")

    all_passed = True

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        icon = "+" if result.passed else "!"

        if not result.passed:
            all_passed = False

        print(f"[{icon}] {result.check_name}: {status}")
        print(f"    {result.message}")

    print(f"{'='*60}")

    if all_passed:
        print("All validation checks passed.")
    else:
        print("Some checks failed - investigate manually (do NOT auto-fix).")

    return all_passed


if __name__ == "__main__":
    # Example usage with a sample ratings file
    import argparse

    parser = argparse.ArgumentParser(description="Validate CFB Power Ratings")
    parser.add_argument("--ratings-file", type=str, required=True,
                        help="Path to ratings CSV file")
    parser.add_argument("--season", type=int, required=True,
                        help="Season year for context-aware checks")

    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings_file)
    results = validate_ratings(ratings, args.season)
    print_validation_report(results, args.season)
