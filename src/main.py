def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print("  GITHUB ACTIVITY LOG – ML DATA PREPARATION PIPELINE (Phase I)")
    print(f"  Root dir : {ROOT_DIR}")
    print(f"  Input    : {DATA_PATH}")
    print(f"  Output   : {OUTPUT_CSV}")
    print("=" * 68)

    try:
        # ══════════════════════════════════════════════════════════
        # STEP 1 – DATA COLLECTION
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 1 · Data Collection")
        df = load_dataset(DATA_PATH)
        if df is None:
            print("  ✗ Dataset not loaded – aborting.")
            return

        print(f"\n  Dataset overview:")
        print(f"    Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"\n  First 3 rows (first 10 cols):")
        print(df.iloc[:3, :10].to_string())

        sub("Defining data types…")
        df = define_data_types(df, "github")
        print(f"  Step 1 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 2 – DATA QUALITY ASSESSMENT
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 2 · Data Quality Assessment")
        quality_report = check_quality(df, "github")
        print_quality_report(quality_report)
        print(f"\n  Step 2 done  ({time.time() - t0:.1f}s)")
        
        # ══════════════════════════════════════════════════════════
        # STEP 3 – MISSING VALUES ANALYSIS & HANDLING
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 3 · Missing Values Analysis")
        missing_before = identify_missing(df)
        total_before   = sum(missing_before.values())
        print(f"\n  Total missing values (before): {total_before:,}")

        sub("Handling missing values (drop >50%, fill Unknown / median)…")
        df = handle_missing_values(df, "github")
        missing_after = identify_missing(df)
        total_after   = sum(missing_after.values())
        print(f"  Total missing values (after) : {total_after:,}")
        print(f"  Step 3 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 4 – DATA CLEANING
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 4 · Data Cleaning")
        df = clean_data(df, "github")
        print(f"  Shape after cleaning: {df.shape}")
        print(f"  Step 4 done  ({time.time() - t0:.1f}s)")

