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
