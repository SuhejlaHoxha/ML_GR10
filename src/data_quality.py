import pandas as pd
import numpy as np


def check_quality(df: pd.DataFrame, dataset_name: str = "github") -> dict:
    report: dict = {
        "dataset"      : dataset_name,
        "rows"         : len(df),
        "columns"      : len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates"   : int(df.duplicated().sum()),
        "numeric_stats": df.select_dtypes(include="number").describe().to_dict(),
    }

    # Duplicate event IDs
    if "_document_id" in df.columns:
        report["duplicate_event_ids"] = int(df["_document_id"].duplicated().sum())

    # actor_is_bot completeness
    if "actor_is_bot" in df.columns:
        total  = len(df)
        filled = int(df["actor_is_bot"].notna().sum())
        report["bot_flag_completeness"] = {
            "filled"    : filled,
            "missing"   : total - filled,
            "pct_filled": round(filled / total * 100, 2),
        }

    # Temporal range of events
    ts_col = "@timestamp" if "@timestamp" in df.columns else None
    if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        report["temporal_range"] = {
            "earliest": str(df[ts_col].min()),
            "latest"  : str(df[ts_col].max()),
            "span_days": int((df[ts_col].max() - df[ts_col].min()).days),
        }

    # Action-type distribution
    if "action" in df.columns:
        top_actions = df["action"].value_counts().head(10).to_dict()
        report["action_cardinality"] = {
            "unique_actions": int(df["action"].nunique()),
            "top_10"        : top_actions,
        }

    # Overall missing-rate
    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isnull().sum().sum())
    report["missing_rate_pct"] = round(total_missing / total_cells * 100, 2)

    return report


def print_quality_report(report: dict) -> None:
    """Pretty-print a quality report returned by check_quality()."""
    print(f"\n  Quality Report – '{report['dataset']}'")
    print(f"    Rows                : {report['rows']:,}")
    print(f"    Columns             : {report['columns']:,}")
    print(f"    Overall missing     : {report['missing_rate_pct']:.2f}% of all cells")
    print(f"    Duplicate rows      : {report['duplicates']:,}")

    if "duplicate_event_ids" in report:
        print(f"    Duplicate event IDs : {report['duplicate_event_ids']:,}")

    if "bot_flag_completeness" in report:
        bc = report["bot_flag_completeness"]
        print(f"    actor_is_bot filled : {bc['filled']:,} / {report['rows']:,}"
              f"  ({bc['pct_filled']}%)")

    if "temporal_range" in report:
        tr = report["temporal_range"]
        print(f"    Event time range    : {tr['earliest']}  →  {tr['latest']}"
              f"  ({tr['span_days']} days)")

    if "action_cardinality" in report:
        ac = report["action_cardinality"]
        print(f"    Unique action types : {ac['unique_actions']:,}")
        print("    Top 5 actions:")
        for action, cnt in list(ac["top_10"].items())[:5]:
            print(f"      {action:<55} {cnt:>5}")
