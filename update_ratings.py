#!/usr/bin/env python3
"""
Update CFB Power Ratings

This script updates ratings for the specified season(s).
Can be run manually or via GitHub Actions.

Usage:
    python update_ratings.py --season 2024
    python update_ratings.py --season 2024 --api-key YOUR_KEY
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

from power_rating import RatingConfig, calculate_ratings, save_ratings


def load_api_key() -> str:
    """Load API key from environment or config file."""
    # Try environment variable first
    api_key = os.environ.get("CFB_API_KEY", "")

    if not api_key:
        # Try config.py
        config_path = Path(__file__).parent / "config.py"
        if config_path.exists():
            config_vars = {}
            exec(open(config_path).read(), config_vars)
            api_key = config_vars.get("CFB_API_KEY", "")

    return api_key


def main():
    parser = argparse.ArgumentParser(description="Update CFB Power Ratings")
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Season year (e.g., 2024 for 2024 season)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="College Football Data API key (or set CFB_API_KEY env var)"
    )

    args = parser.parse_args()

    # Load configuration
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
        # Calculate ratings
        ratings = calculate_ratings(args.season, config)

        # Save to CSV
        output_path = save_ratings(ratings, args.season, config)

        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
              f"Saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        raise

    print(f"\nAll 1 season(s) updated successfully.")


if __name__ == "__main__":
    main()
