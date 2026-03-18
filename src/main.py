import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collection      import load_dataset, define_data_types

ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH       = os.path.join(ROOT_DIR, "unprocessed dataset", "github.csv")
PROCESSED_DIR   = os.path.join(ROOT_DIR, "processed dataset")
EDA_PLOTS_DIR   = os.path.join(ROOT_DIR, "eda_plots")
OUTPUT_CSV      = os.path.join(PROCESSED_DIR, "processed_github.csv")

TARGET_COL      = "actor_is_bot"
SAMPLE_FRAC     = 0.80         
RANDOM_SEED     = 42


FINAL_FEATURES  = [
    "action", "actor", "actor_id",
    "actor_is_bot",               
    "operation_type", "visibility",
    "repo", "repo_id", "org", "org_id",
    "user_agent", "is_robot",
    "programmatic_access_type",
    "request_category",
    "event_hour", "event_dayofweek",
    "event_month", "event_year", "is_weekend",
    "actor_event_count", "actor_bot_ratio",
    "actor_event_velocity",
    "is_programmatic", "is_integration_event",
    "is_outlier",
]


def header(title: str) -> None:
    width = 68
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def sub(msg: str) -> None:
    print(f"\n  ► {msg}")



def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print("  GITHUB ACTIVITY LOG – ML DATA PREPARATION PIPELINE (Phase I)")
    print(f"  Root dir : {ROOT_DIR}")
    print(f"  Input    : {DATA_PATH}")
    print(f"  Output   : {OUTPUT_CSV}")
    print("=" * 68)
