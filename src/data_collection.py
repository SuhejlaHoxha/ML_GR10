import pandas as pd
from pathlib import Path

TIMESTAMP_COLS   = ["@timestamp"]    
DATETIME_COLS    = ["created_at", "updated_at", "deleted_at",
                    "started_at", "completed_at", "cancelled_at"]
IDENTIFIER_COLS  = ["_document_id", "actor_id", "repo_id", "org_id",
                    "user_id", "business_id", "request_id",
                    "pull_request_id", "workflow_id", "server_id",
                    "integration_id", "hook_id", "alert_id"]
CATEGORY_COLS    = ["action", "operation_type", "visibility",
                    "request_category", "category_type",
                    "programmatic_access_type", "method",
                    "actor_location.country_code"]
BOOLEAN_COLS     = ["actor_is_bot", "is_robot", "public_repo",
                    "actor_is_agent", "read_only", "active",
                    "prerelease", "enabled"]


def load_dataset(file_path: str) -> pd.DataFrame | None:
    path = Path(file_path)
    if not path.exists():
        print(f"  File not found: {path}")
        return None

    print(f"  Loading: {path.name}")
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"  Loaded {path.name}: {len(df):,} rows × {len(df.columns)} columns")
        return df
    except Exception as exc:
        print(f"  Error loading {path.name}: {exc}")
        return None


def define_data_types(df: pd.DataFrame, dataset_name: str = "github") -> pd.DataFrame:
    print(f"\n  Defining data types for '{dataset_name}' ({len(df):,} rows)…")

    for col in TIMESTAMP_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
            print(f"    → {col}: epoch-ms → datetime")

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            print(f"    → {col}: str → datetime")

    for col in IDENTIFIER_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in CATEGORY_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
            print(f"    → {col}: → category ({df[col].nunique()} unique)")

    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n  '{dataset_name}' shape after type definition : {df.shape}")
    print("  Column-type summary:")
    print("  " + df.dtypes.value_counts().to_string().replace("\n", "\n  "))
    print("  " + "-" * 56)
    return df
