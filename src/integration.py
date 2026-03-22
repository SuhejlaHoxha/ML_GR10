import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# DATASET INTEGRATION NOTE
# ─────────────────────────────────────────────────────────────────────
def dataset_integration_note() -> None:
    print("""
  ══════════════════════════════════════════════════════════════════
  DATASET INTEGRATION
  ══════════════════════════════════════════════════════════════════
  Only a single dataset (github.csv) is provided for this project.
  Dataset merging / joining is therefore NOT required.
  ══════════════════════════════════════════════════════════════════
""")


# ─────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────
def aggregate_by_actor(df: pd.DataFrame) -> pd.DataFrame:

    actor_col = "actor" if "actor" in df.columns else None
    if actor_col is None:
        print("actor column not found – skipping actor aggregation.")
        return pd.DataFrame()

    agg_parts = {actor_col: "Actor event counts"}
    agg_dict: dict = {"@timestamp" if "@timestamp" in df.columns else df.columns[0]: "count"}

    if "actor_is_bot" in df.columns:
        agg_dict["actor_is_bot"] = ["sum", "mean"]

    result = (
        df.groupby(actor_col, observed=True)
          .agg(agg_dict)
          .reset_index()
    )
    result.columns = [
        "_".join(filter(None, col)).strip("_") if isinstance(col, tuple) else col
        for col in result.columns
    ]
    result.rename(columns={result.columns[1]: "event_count"}, inplace=True)
    result = result.sort_values("event_count", ascending=False).reset_index(drop=True)
    return result


def aggregate_by_action(df: pd.DataFrame) -> pd.DataFrame:
    col = "action" if "action" in df.columns else None
    if col is None:
        print("action column not found – skipping action aggregation.")
        return pd.DataFrame()

    result = (
        df.groupby(col, observed=True)
          .size()
          .reset_index(name="event_count")
          .sort_values("event_count", ascending=False)
          .reset_index(drop=True)
    )
    return result


def aggregate_by_time(df: pd.DataFrame,
                       ts_col: str = "@timestamp",
                       freq: str = "h") -> pd.DataFrame:
 
    if ts_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
        print(f"{ts_col} is not a datetime column – skipping time aggregation.")
        return pd.DataFrame()

    result = (
        df.resample(freq, on=ts_col)
          .size()
          .reset_index(name="event_count")
    )
    return result


def aggregate_by_org(df: pd.DataFrame) -> pd.DataFrame:
    col = "org" if "org" in df.columns else None
    if col is None:
        print("org column not found – skipping org aggregation.")
        return pd.DataFrame()

    result = (
        df.groupby(col, observed=True)
          .size()
          .reset_index(name="event_count")
          .sort_values("event_count", ascending=False)
          .reset_index(drop=True)
    )
    return result


# ─────────────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────────────
def sample_data(df: pd.DataFrame,
                n_samples: int,
                method: str = "random",
                stratify_col: str = "actor_is_bot") -> pd.DataFrame:

    n_samples = min(n_samples, len(df))

    try:
        if method == "stratified" and stratify_col in df.columns:
            sampled = (
                df.groupby(stratify_col, observed=True)
                  .apply(lambda g: g.sample(frac=n_samples / len(df),
                                            random_state=42))
                  .reset_index(drop=True)
            )
            print(f"  Stratified sample by '{stratify_col}': {len(sampled):,} rows")
            return sampled
        else:
            if method == "stratified":
                print(f"{stratify_col} not found – falling back to random sampling.")
            sampled = df.sample(n=n_samples, random_state=42)
            print(f"  Random sample: {len(sampled):,} rows")
            return sampled.reset_index(drop=True)
    except Exception as exc:
        print(f"  Sampling error: {exc} – returning original dataframe.")
        return df
