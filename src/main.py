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

        # ══════════════════════════════════════════════════════════
        # STEP 5 – DATASET INTEGRATION & AGGREGATION
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 5 · Dataset Integration & Aggregation")

        dataset_integration_note()

        sub("Aggregation 1 – events per actor (top 10):")
        actor_agg = aggregate_by_actor(df)
        if not actor_agg.empty:
            print(actor_agg.head(10).to_string(index=False))

        sub("Aggregation 2 – events per action type (top 10):")
        action_agg = aggregate_by_action(df)
        if not action_agg.empty:
            print(action_agg.head(10).to_string(index=False))

        sub("Aggregation 3 – events per organisation (top 10):")
        org_agg = aggregate_by_org(df)
        if not org_agg.empty:
            print(org_agg.head(10).to_string(index=False))

        ts_col = "@timestamp" if "@timestamp" in df.columns else None
        if ts_col and pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            sub("Aggregation 4 – hourly event volume (first 10 hours):")
            time_agg = aggregate_by_time(df, ts_col, freq="h")
            if not time_agg.empty:
                print(time_agg.head(10).to_string(index=False))

        print(f"\n  Step 5 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 6 – SAMPLING
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 6 · Sampling")
        n_sample = int(len(df) * SAMPLE_FRAC)
        sub(f"Drawing {int(SAMPLE_FRAC*100)}% stratified sample "
            f"({n_sample:,} rows)…")
        df = sample_data(df, n_samples=n_sample,
                          method="stratified",
                          stratify_col=TARGET_COL)
        print(f"  Shape after sampling: {df.shape}")
        print(f"  Step 6 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 7 – ADVANCED PREPROCESSING
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 7 · Advanced Preprocessing")
        preprocessor   = AdvancedPreprocessor()
        original_df    = df.copy()

        sub("7.1 – Creating derived features…")
        df = preprocessor.create_derived_features(df)

        sub("7.2 – Label encoding categorical columns…")
        df = preprocessor.encode_categoricals(df)

        sub("7.3 – Discretisation & binarisation…")
        try:
            df = preprocessor.discretize_and_binarize(df)
            print(f"  Shape after discretisation: {df.shape}")
        except Exception as exc:
            print(f"  ⚠  Discretisation error: {exc}")

        sub("7.4 – Data transformations…")
        df = preprocessor.apply_transformations(df)

        sub("7.5 – Dimension reduction (PCA, 95% variance)…")
        try:
            df_pca = preprocessor.dimension_reduction(
                df,
                target_col=TARGET_COL,
                method="pca",
                n_components=0.95,
                feature_types="numeric",
            )
        except Exception as exc:
            print(f"  ⚠  PCA error: {exc}")
            df_pca = df.copy()

        sub("7.5 – Dimension reduction (Univariate, top-20 features)…")
        try:
            df_selected = preprocessor.dimension_reduction(
                df,
                target_col=TARGET_COL,
                method="univariate",
                n_components=20,
                feature_types="numeric",
            )
        except Exception as exc:
            print(f"  ⚠  Univariate selection error: {exc}")
            df_selected = df.copy()

        summary = preprocessor.get_preprocessing_summary(original_df, df)
        print(f"\n  Step 7 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 8 – EXPLORATORY DATA ANALYSIS
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 8 · Exploratory Data Analysis (EDA)")
        eda = EDAAnalyzer(save_plots=True, output_dir=EDA_PLOTS_DIR)

        num_cols = [c for c in df.select_dtypes(include="number").columns
                    if not c.startswith("outlier_")]
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"  Numeric columns  : {len(num_cols)}")
        print(f"  Categorical cols : {len(cat_cols)}")

        # 8.1 Summary statistics
        sub("8.1 – Numerical summary (first 30 cols)…")
        try:
            num_summary = eda.numerical_summary(df, num_cols[:30])
            print(num_summary.head(10).to_string())
            num_summary.to_csv(os.path.join(EDA_PLOTS_DIR, "numerical_summary.csv"))
        except Exception as exc:
            print(f"  ⚠  Numerical summary error: {exc}")

        sub("8.2 – Categorical summary…")
        try:
            cat_summary = eda.categorical_summary(df, cat_cols[:15])
            for col, counts in list(cat_summary.items())[:3]:
                print(f"\n  {col}:")
                print(counts.head(8).to_string())
        except Exception as exc:
            print(f"  ⚠  Categorical summary error: {exc}")

        # 8.3 Distribution + boxplots
        sub("8.3 – Distribution & boxplots (top 5 numeric cols)…")
        try:
            plot_cols = num_cols[:5]
            eda.distribution_plots(df, plot_cols)
            eda.boxplot(df, plot_cols)
        except Exception as exc:
            print(f"  ⚠  Distribution/boxplot error: {exc}")

        # 8.4 Correlation matrix
        sub("8.4 – Correlation matrix…")
        try:
            corr_cols = [c for c in num_cols[:25] if df[c].var() > 0]
            if len(corr_cols) > 1:
                corr = eda.correlation_matrix(df, columns=corr_cols, figsize=(14, 12))
                corr.to_csv(os.path.join(EDA_PLOTS_DIR, "correlation_matrix.csv"))

                # Top correlations
                mask  = np.triu(np.ones_like(corr, dtype=bool), k=1)
                pairs = [
                    (corr.columns[i], corr.columns[j], corr.iloc[i, j])
                    for i in range(len(corr.columns))
                    for j in range(i+1, len(corr.columns))
                    if not pd.isna(corr.iloc[i, j])
                ]
                pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                print("  Top 10 strongest correlations:")
                for c1, c2, v in pairs[:10]:
                    print(f"    {c1:<35} ↔ {c2:<35} {v:+.3f}")
        except Exception as exc:
            print(f"  ⚠  Correlation error: {exc}")

        # 8.5 PCA analysis
        sub("8.5 – PCA analysis…")
        try:
            pca_cols = [c for c in num_cols[:30]
                        if df[c].var() > 0 and df[c].notna().any()]
            if len(pca_cols) >= 2:
                n_pca = min(10, len(pca_cols))
                pca_df_eda, ev = eda.pca_analysis(df, pca_cols, n_components=n_pca)
                print(f"  PCA ({n_pca} components) – explained variance:")
                for i, v in enumerate(ev):
                    print(f"    PC{i+1}: {v:.2%}  (cumulative: {ev[:i+1].sum():.2%})")
                labels = df[TARGET_COL].values if TARGET_COL in df.columns else None
                eda.plot_pca(pca_df_eda.iloc[:, :2], labels=labels)
                pca_df_eda.to_csv(
                    os.path.join(EDA_PLOTS_DIR, "pca_components.csv"), index=False
                )
        except Exception as exc:
            print(f"  ⚠  PCA EDA error: {exc}")

        # 8.6 Pairplot
        sub("8.6 – Pairplot…")
        try:
            pair_cols = [c for c in num_cols[:8] if df[c].var() > 0][:5]
            if len(pair_cols) >= 2:
                eda.pairplot(df, pair_cols)
        except Exception as exc:
            print(f"  ⚠  Pairplot error: {exc}")

        # 8.7 GitHub-specific plots
        sub("8.7 – GitHub-specific charts…")
        try:
            eda.plot_class_distribution(df, TARGET_COL)
            eda.plot_events_by_hour(df)
            eda.plot_events_by_dayofweek(df)
            eda.plot_top_actions(df)
        except Exception as exc:
            print(f"  ⚠  GitHub-specific chart error: {exc}")

        # 8.8 Grouped summaries
        sub("8.8 – Grouped summaries…")
        try:
            if "actor" in df.columns and "actor_event_count" in df.columns:
                gs = eda.grouped_summary(df, "actor", "actor_event_count")
                print("  actor_event_count by actor (top 10):")
                print(gs.head(10).to_string())
        except Exception as exc:
            print(f"  ⚠  Grouped summary error: {exc}")

        print(f"\n  EDA plots saved → {EDA_PLOTS_DIR}")
        plots = eda.list_saved_plots()
        print(f"  Plots generated  : {len(plots)}")
        for p in plots:
            print(f"    {p}")
        print(f"\n  Step 8 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 9 – CLASS IMBALANCE & RESAMPLING
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 9 · Class Imbalance Detection & Resampling (SMOTE / ADASYN)")

        if TARGET_COL in df.columns:
            counts = df[TARGET_COL].value_counts().sort_index()
            print(f"\n  Target: '{TARGET_COL}'")
            print(f"  {'Class':<10} {'Count':>8}  {'%':>8}")
            print(f"  {'-'*30}")
            for cls, cnt in counts.items():
                print(f"  {cls!s:<10} {cnt:>8,}  {cnt/len(df)*100:>7.2f}%")
            ratio = counts.max() / max(counts.min(), 1)
            print(f"\n  Imbalance ratio ≈ {ratio:.1f} : 1")
            print("  ⚠  Imbalance detected → applying SMOTE and ADASYN.")

            sub("Applying SMOTE and ADASYN…")
            df_smote, df_adasyn = apply_resampling(
                df, target_col=TARGET_COL, seed=RANDOM_SEED
            )

            # Before/after comparison chart
            before_dist   = df[TARGET_COL].value_counts().to_dict()
            smote_dist    = df_smote[TARGET_COL].value_counts().to_dict()
            adasyn_dist   = df_adasyn[TARGET_COL].value_counts().to_dict()
            eda.plot_resampling_comparison(before_dist, smote_dist, adasyn_dist)
        else:
            print(f"  ⚠  Target column '{TARGET_COL}' not found.")

        print(f"\n  Step 9 done  ({time.time() - t0:.1f}s)")

        # ══════════════════════════════════════════════════════════
        # STEP 10 – OUTLIER DETECTION
        # ══════════════════════════════════════════════════════════
        t0 = time.time()
        header("STEP 10 · Outlier Detection")
        detector = OutlierDetector()

        # Select meaningful numeric columns
        exclude_ids   = {"_document_id", "actor_id", "repo_id",
                         "org_id", "user_id", "business_id",
                         "request_id", "pull_request_id",
                         "workflow_id", "server_id"}
        all_num       = df.select_dtypes(include="number").columns.tolist()
        meaningful    = [c for c in all_num
                         if c not in exclude_ids
                         and not c.startswith("outlier_")]

        
        key_cols = [c for c in meaningful
                    if any(kw in c.lower()
                           for kw in ["count", "velocity", "ratio",
                                      "hour", "bot", "score",
                                      "event", "active"])]
        if len(key_cols) > 15:
            key_cols = sorted(key_cols,
                              key=lambda c: df[c].var(),
                              reverse=True)[:15]

        print(f"  Key columns for IQR / Z-Score: {len(key_cols)}")

        sub("10.1 – IQR outlier detection…")
        try:
            df = detector.detect_iqr(df, key_cols)
            iqr_total = sum(v for k, v in detector.get_summary().items()
                            if k.startswith("iqr_"))
            print(f"  Total IQR outliers: {iqr_total:,}")
        except Exception as exc:
            print(f"  ⚠  IQR error: {exc}")

        sub("10.2 – Z-Score outlier detection…")
        try:
            df = detector.detect_zscore(df, key_cols, threshold=3)
            zs_total = sum(v for k, v in detector.get_summary().items()
                           if k.startswith("zscore_"))
            print(f"  Total Z-Score outliers: {zs_total:,}")
        except Exception as exc:
            print(f"  ⚠  Z-Score error: {exc}")

        
        feat_cols = [c for c in meaningful if df[c].var() > 0][:50]

        sub("10.3 – Isolation Forest…")
        try:
            df = detector.detect_isolation_forest(df, feat_cols, contamination=0.05)
        except Exception as exc:
            print(f"  ⚠  Isolation Forest error: {exc}")

        sub("10.4 – Local Outlier Factor…")
        try:
            df = detector.detect_lof(df, feat_cols, contamination=0.05)
        except Exception as exc:
            print(f"  ⚠  LOF error: {exc}")

        sub("10.5 – Mahalanobis distance (on PCA components)…")
        try:
            pca_cols = [c for c in df.columns if c.startswith("PC_")]
            if len(pca_cols) >= 2:
                df = detector.detect_mahalanobis(df, pca_cols[:10], threshold=3.5)
            else:
                print("  ⚠  No PCA components found – skipping Mahalanobis.")
        except Exception as exc:
            print(f"  ⚠  Mahalanobis error: {exc}")

        sub("10.6 – Rare-category detection…")
        try:
            for col in ["action", "operation_type"]:
                if col in df.columns:
                    df = detector.detect_rare_categories(df, col, min_freq=0.01)
        except Exception as exc:
            print(f"  ⚠  Rare-category error: {exc}")

        sub("10.6 – Combined outlier score…")
        try:
            df = detector.compute_outlier_score(df)
            df["outlier_type"] = df["outlier_score"].apply(
                detector.map_outlier_type
            )
            type_counts = df["outlier_type"].value_counts()
            print("  Outlier type distribution:")
            for t, c in type_counts.items():
                print(f"    {t:<10} {c:>6,}  ({c/len(df)*100:.1f}%)")
        except Exception as exc:
            print(f"  ⚠  Outlier score error: {exc}")

        sub("10.7 – Filtering false detections (≥2 methods must agree)…")
        try:
            df = detector.validate_outliers(df, min_agreement=2,
                                             use_multivariate=True)
            df = detector.filter_false_detections(df, method="agreement",
                                                   min_agreement=2)
            report = detector.get_false_detection_report(df)
            print("  False-detection report:")
            for k, v in report.items():
                if k != "method_contributions":
                    print(f"    {k:<35} {v}")
        except Exception as exc:
            print(f"  ⚠  False-detection filter error: {exc}")
            traceback.print_exc()


        if "outlier_confirmed" in df.columns:
            df["is_outlier"] = df["outlier_confirmed"].astype(int)
        elif "outlier_score" in df.columns:
            df["is_outlier"] = (df["outlier_score"] > 0).astype(int)
        else:
            df["is_outlier"] = 0

        print(f"\n  Dataset shape after outlier detection: {df.shape}")
        print(f"  Step 10 done  ({time.time() - t0:.1f}s)")


        t0 = time.time()
        header("STEP 11 · Subset Selection – Final Feature Set")

        selected = [c for c in FINAL_FEATURES if c in df.columns]

        if TARGET_COL not in selected and TARGET_COL in df.columns:
            selected.insert(0, TARGET_COL)

        df_final = df[selected].copy()
        print(f"  Selected {len(selected)} features:")
        for f in selected:
            print(f"    • {f}")
        print(f"  Final subset shape: {df_final.shape}")
        print(f"  Step 11 done  ({time.time() - t0:.1f}s)")


        t0 = time.time()
        header("STEP 12 · Save Processed Dataset")

        if TARGET_COL in df_final.columns:
            before = len(df_final)
            df_final = df_final.dropna(subset=[TARGET_COL])
            print(f"  Dropped {before - len(df_final):,} rows with missing target.")

        df_final = df_final.reset_index(drop=True)


        remaining_nan = df_final.isnull().sum().sum()
        print(f"  Remaining NaN values : {remaining_nan}")


        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\n  ✓ Processed dataset saved → {OUTPUT_CSV}")
        print(f"  Final shape : {df_final.shape[0]:,} rows × "
              f"{df_final.shape[1]} columns")
        if TARGET_COL in df_final.columns:
            print(f"  Target distribution:")
            for cls, cnt in df_final[TARGET_COL].value_counts().items():
                print(f"    {cls} → {cnt:,}  ({cnt/len(df_final)*100:.1f}%)")
        print(f"\n  Step 12 done  ({time.time() - t0:.1f}s)")


        header("PIPELINE COMPLETE")
        print(f"""
  Output files
  ───────────────────────────────────────────────────────────────
  Processed CSV : {OUTPUT_CSV}
  EDA plots     : {EDA_PLOTS_DIR}/
  ───────────────────────────────────────────────────────────────
  Final dataset : {df_final.shape[0]:,} rows × {df_final.shape[1]} columns
  Ready for Phase II (model selection & training).
""")

    except Exception as exc:
        header("ERROR OCCURRED")
        print(f"  Exception: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
