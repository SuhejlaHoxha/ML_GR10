import pandas as pd
from sklearn.impute import SimpleImputer

MISSING_DROP_THRESHOLD = 0.50


def clean_data(df: pd.DataFrame, dataset_name: str = "github") -> pd.DataFrame:
    print(f"  Cleaning '{dataset_name}'...")

    # _document_id is only safe to deduplicate on when it is truly unique.
    id_col = "_document_id"
    if id_col in df.columns:
        total_rows = len(df)
        unique_ids = df[id_col].nunique(dropna=False)
        duplicate_ids = total_rows - unique_ids

        if duplicate_ids == 0:
            print("    _document_id is unique across all rows.")
        else:
            print(f"    Skipping _document_id deduplication: "
                  f"{duplicate_ids:,} repeated IDs found "
                  f"({total_rows:,} rows, {unique_ids:,} unique IDs).")
            print("      _document_id is not a reliable event key here, "
                  "so dropping on it would discard valid rows.")

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"    Full-row duplicates removed : {removed:,}")

    df.columns = (
        df.columns
          .str.replace(".", "_", regex=False)
          .str.replace(" ", "_", regex=False)
          .str.lower()
          .str.strip()
    )
    print("    Column names standardised  (dots / spaces -> underscores, lowercase).")

    return df.reset_index(drop=True)


def identify_missing(df: pd.DataFrame) -> dict:
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    total = sum(missing.values())

    print(f"  Missing-value summary: {total:,} NaNs across {len(missing)} columns")
    if missing:
        top10 = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]
        print("  Top 10 columns by NaN count:")
        for col, cnt in top10:
            pct = cnt / len(df) * 100
            print(f"    {col:<55} {cnt:>7,}  ({pct:.1f}%)")
    return missing


def handle_missing_values(df: pd.DataFrame,
                          dataset_name: str = "github") -> pd.DataFrame:
    print(f"  Handling missing values for '{dataset_name}'...")
    before_cols = df.shape[1]

    null_rates = df.isnull().mean()
    drop_cols = null_rates[null_rates > MISSING_DROP_THRESHOLD].index.tolist()
    df = df.drop(columns=drop_cols)
    print(f"    Dropped {len(drop_cols)} columns with >{int(MISSING_DROP_THRESHOLD*100)}% "
          f"missing  ->  {df.shape[1]} columns remain.")

    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        if df[col].isnull().any():
            if hasattr(df[col], "cat"):
                if "Unknown" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(["Unknown"])
            df[col] = df[col].fillna("Unknown")
    print(f"    Object/category NaNs -> 'Unknown'  ({len(obj_cols)} columns).")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    valid_num_cols = [c for c in num_cols if df[c].notna().any()]
    all_nan_cols = [c for c in num_cols if df[c].isna().all()]

    if valid_num_cols:
        imputer = SimpleImputer(strategy="median")
        imputed = imputer.fit_transform(df[valid_num_cols])
        df[valid_num_cols] = pd.DataFrame(
            imputed,
            columns=valid_num_cols,
            index=df.index,
        )

    for col in all_nan_cols:
        df[col] = df[col].fillna(0)

    print(f"    Numeric NaNs -> column median  ({len(valid_num_cols)} columns).")
    if all_nan_cols:
        print(f"    All-NaN numeric columns -> 0  ({len(all_nan_cols)} columns).")

    remaining = int(df.isnull().sum().sum())
    print(f"    Remaining NaNs after handling : {remaining}")
    print(f"    Columns: {before_cols} -> {df.shape[1]}")
    return df
