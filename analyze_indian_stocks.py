from pathlib import Path
import warnings
import numpy as np
import pandas as pd

DATA_FOLDER = Path(r"PUT_EXTRACTED_DATASET_FOLDER_HERE")
MODEL_STOCK_COUNT = 30
HORIZON = 20
FEATURES = ["ret_5", "ret_20", "ret_60", "vol_20", "volume_change_5"]
ALPHAS = [0.1, 1.0, 10.0, 100.0]


def get_data_folder():
    if "PUT_EXTRACTED_DATASET_FOLDER_HERE" in str(DATA_FOLDER):
        return Path(__file__).resolve().parent
    return DATA_FOLDER.expanduser().resolve()


def find_column(df, names):
    available = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in available:
            return available[name.lower()]
    return None


def read_stock(path):
    try:
        raw = pd.read_csv(path, low_memory=False)
        date_col = find_column(raw, ["Date", "Datetime", "timestamp"])
        price_col = find_column(
            raw, ["Adj Close", "Adjusted Close", "Adj_Close", "Close"]
        )
        volume_col = find_column(
            raw, ["Volume", "Tottrdqty", "TotalTradedQuantity"]
        )

        if date_col is None or price_col is None:
            return None

        dates = pd.to_datetime(raw[date_col], errors="coerce")
        prices = pd.to_numeric(raw[price_col], errors="coerce")

        if volume_col is not None:
            volumes = pd.to_numeric(raw[volume_col], errors="coerce")
        else:
            volumes = pd.Series(0.0, index=raw.index)

        stock = pd.DataFrame({
            "Date": dates,
            "Price": prices,
            "Volume": volumes,
        })
        stock = stock.dropna(subset=["Date", "Price"])
        stock = stock[stock["Price"] > 0]
        stock = stock.sort_values("Date").drop_duplicates(
            "Date", keep="last"
        )

        if len(stock) < 100:
            return None

        stock["Ticker"] = path.stem
        return stock.reset_index(drop=True)

    except Exception as error:
        print(f"Skipped {path.name}: {error}")
        return None


def make_features(stock):
    frame = stock.copy()
    daily = frame["Price"].pct_change()

    frame["ret_5"] = frame["Price"].pct_change(5)
    frame["ret_20"] = frame["Price"].pct_change(20)
    frame["ret_60"] = frame["Price"].pct_change(60)
    frame["vol_20"] = daily.rolling(20).std() * np.sqrt(252)
    frame["volume_change_5"] = (
        np.log1p(frame["Volume"].clip(lower=0)).diff(5)
    )
    frame["target_20d"] = (
        frame["Price"].shift(-HORIZON) / frame["Price"] - 1
    )

    return frame.replace([np.inf, -np.inf], np.nan)


def fit_ridge(x, y, alpha):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0

    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])

    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0

    weights = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ y,
    )

    return mean, scale, weights


def predict_ridge(model, x):
    mean, scale, weights = model
    z = (np.asarray(x, dtype=float) - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    return design @ weights


def main():
    data_folder = get_data_folder()
    files = sorted(
        path for path in data_folder.rglob("*.csv")
        if "project_results" not in path.parts
    )

    if not files:
        raise SystemExit(
            f"No CSV files found under {data_folder}. Check DATA_FOLDER."
        )

    output_folder = Path(__file__).resolve().parent / "project_results"
    output_folder.mkdir(exist_ok=True)

    print(f"Looking in: {data_folder}")
    print(f"Found {len(files)} CSV files. Reading them now...")

    company_rows = []
    annual_rows = []
    skipped = []
    market_by_date = {}
    usable_files = []
    excluded_moves = 0

    for number, path in enumerate(files, start=1):
        stock = read_stock(path)

        if stock is None:
            skipped.append({
                "File": str(path.relative_to(data_folder)),
                "Reason": (
                    "No usable date/price columns or fewer than 100 rows"
                ),
            })
            continue

        usable_files.append((len(stock), path))

        daily = stock["Price"].pct_change()
        clean = daily.where(daily.between(-0.80, 2.00))
        excluded_moves += int(
            (daily.notna() & clean.isna()).sum()
        )

        valid = clean.dropna()
        wealth = (1 + valid).cumprod()
        drawdown = wealth / wealth.cummax() - 1

        company_rows.append({
            "Stock file": path.stem,
            "Start date": stock["Date"].min().date(),
            "End date": stock["Date"].max().date(),
            "Trading rows": len(stock),
            "Total return (filtered daily moves)": float(
                (1 + valid).prod() - 1
            ),
            "Maximum drawdown": (
                float(drawdown.min()) if len(drawdown) else np.nan
            ),
            "Up-day share": (
                float((valid > 0).mean()) if len(valid) else np.nan
            ),
        })

        for date, value in zip(stock["Date"], clean):
            if pd.isna(value):
                continue

            key = pd.Timestamp(date).normalize()
            agg = market_by_date.setdefault(key, [0.0, 0, 0, 0])
            agg[0] += float(value)
            agg[1] += 1
            agg[2] += int(value > 0)
            agg[3] += int(value < 0)

        yearly_prices = (
            stock.assign(Year=stock["Date"].dt.year)
            .groupby("Year")["Price"]
            .last()
        )
        yearly = yearly_prices.pct_change()

        if len(yearly):
            first_year = yearly.index[0]
            first_year_start = stock.loc[
                stock["Date"].dt.year == first_year, "Price"
            ].iloc[0]
            yearly.loc[first_year] = (
                yearly_prices.iloc[0] / first_year_start - 1
            )

        for year, value in yearly.items():
            annual_rows.append({
                "Year": int(year),
                "Stock file": path.stem,
                "Return": float(value),
            })

        if number % 100 == 0:
            print(f"Read {number}/{len(files)} files...")

    if not company_rows or not annual_rows:
        raise SystemExit(
            "No usable stock histories or yearly returns were found."
        )

    company_summary = pd.DataFrame(company_rows)
    company_summary = company_summary.sort_values(
        "Trading rows", ascending=False
    )
    annual_returns = pd.DataFrame(annual_rows)

    daily_rows = []
    for date, (total, count, up, down) in sorted(market_by_date.items()):
        daily_rows.append({
            "Date": date,
            "Average stock daily return": total / count,
            "Stocks counted": count,
            "Stocks up": up,
            "Stocks down": down,
            "Share of stocks up": up / count if count else np.nan,
        })

    daily = pd.DataFrame(daily_rows)
    daily["Year"] = daily["Date"].dt.year

    company_year = annual_returns.groupby("Year").agg(**{
        "Median listed-stock return": ("Return", "median"),
        "Share of listed stocks positive": (
            "Return",
            lambda values: (values > 0).mean(),
        ),
        "Number of stock files": ("Stock file", "nunique"),
    })

    breadth_year = daily.groupby("Year").agg(**{
        "Average daily share of stocks up": (
            "Share of stocks up", "mean"
        ),
        "Trading dates": ("Date", "count"),
    })

    market_year = company_year.join(
        breadth_year, how="outer"
    ).reset_index()

    selected = [
        path
        for _, path in sorted(
            usable_files,
            key=lambda item: item[0],
            reverse=True,
        )[:MODEL_STOCK_COUNT]
    ]

    panels = []
    latest_rows = []

    print(
        f"Preparing the forecast model for "
        f"{len(selected)} longest-history files..."
    )

    for path in selected:
        stock = read_stock(path)
        if stock is None:
            continue

        panel = make_features(stock)
        labeled = panel.dropna(
            subset=FEATURES + ["target_20d"]
        )

        if len(labeled):
            panels.append(
                labeled[
                    ["Date", "Ticker"] + FEATURES + ["target_20d"]
                ]
            )

        latest = panel.dropna(subset=FEATURES).tail(1)
        if len(latest):
            latest_rows.append(
                latest[["Date", "Ticker", "Price"] + FEATURES]
            )

    if not panels or not latest_rows:
        raise SystemExit("Not enough history to prepare model rows.")

    data = pd.concat(panels, ignore_index=True)
    data = data.sort_values("Date")

    train = data[data["Date"] < pd.Timestamp("2015-12-03")]
    validation = data[
        (data["Date"] >= pd.Timestamp("2016-01-01"))
        & (data["Date"] < pd.Timestamp("2019-01-01"))
    ]
    test = data[data["Date"] >= pd.Timestamp("2019-01-29")]

    if min(len(train), len(validation), len(test)) == 0:
        raise SystemExit(
            "Data must include training, 2016–2018 validation, "
            "and 2019-or-later test dates."
        )

    x_train = train[FEATURES].to_numpy()
    y_train = train["target_20d"].to_numpy()
    x_valid = validation[FEATURES].to_numpy()
    y_valid = validation["target_20d"].to_numpy()

    scores = []
    for alpha in ALPHAS:
        candidate = fit_ridge(x_train, y_train, alpha)
        valid_prediction = predict_ridge(candidate, x_valid)
        mae = float(np.mean(np.abs(y_valid - valid_prediction)))
        scores.append((mae, alpha))

    _, best_alpha = min(scores)

    model = fit_ridge(x_train, y_train, best_alpha)
    actual = test["target_20d"].to_numpy()
    prediction = predict_ridge(
        model, test[FEATURES].to_numpy()
    )
    zero = np.zeros(len(actual))

    metrics = pd.DataFrame([
        {
            "Method": "Ridge model",
            "Test rows": len(test),
            "MAE": np.mean(np.abs(actual - prediction)),
            "RMSE": np.sqrt(np.mean((actual - prediction) ** 2)),
            "Direction accuracy": np.mean(
                (actual > 0) == (prediction > 0)
            ),
        },
        {
            "Method": "Predict zero return",
            "Test rows": len(test),
            "MAE": np.mean(np.abs(actual - zero)),
            "RMSE": np.sqrt(np.mean((actual - zero) ** 2)),
            "Direction accuracy": np.mean(
                (actual > 0) == (zero > 0)
            ),
        },
        {
            "Method": "Always predict up",
            "Test rows": len(test),
            "MAE": np.nan,
            "RMSE": np.nan,
            "Direction accuracy": np.mean(actual > 0),
        },
    ])

    test_output = test[
        ["Date", "Ticker", "target_20d"]
    ].copy()
    test_output["Predicted 20-session return"] = prediction
    test_output["Actual direction"] = np.where(
        actual > 0, "UP", "DOWN"
    )
    test_output["Predicted direction"] = np.where(
        prediction > 0, "UP", "DOWN"
    )

    final_model = fit_ridge(
        data[FEATURES].to_numpy(),
        data["target_20d"].to_numpy(),
        best_alpha,
    )

    latest = pd.concat(latest_rows, ignore_index=True)
    latest["Predicted next 20-session return"] = predict_ridge(
        final_model,
        latest[FEATURES].to_numpy(),
    )
    latest["Predicted direction"] = np.where(
        latest["Predicted next 20-session return"] > 0,
        "UP",
        "DOWN",
    )
    latest = latest.rename(columns={
        "Date": "Last data date",
        "Price": "Last adjusted close",
    })
    latest = latest[[
        "Ticker",
        "Last data date",
        "Last adjusted close",
        "Predicted next 20-session return",
        "Predicted direction",
    ]]

    best_year = market_year.loc[
        market_year["Median listed-stock return"].idxmax(),
        "Year",
    ]
    worst_year = market_year.loc[
        market_year["Median listed-stock return"].idxmin(),
        "Year",
    ]

    summary = pd.DataFrame([
        {
            "Measure": "Usable company CSV files",
            "Value": len(company_summary),
        },
        {"Measure": "Skipped files", "Value": len(skipped)},
        {
            "Measure": "Data begins",
            "Value": str(daily["Date"].min().date()),
        },
        {
            "Measure": "Data ends",
            "Value": str(daily["Date"].max().date()),
        },
        {
            "Measure": "Extreme daily moves excluded from summaries",
            "Value": excluded_moves,
        },
        {
            "Measure": "Best year by median stock return",
            "Value": int(best_year),
        },
        {
            "Measure": "Worst year by median stock return",
            "Value": int(worst_year),
        },
        {
            "Measure": "Files used in forecast model",
            "Value": len(selected),
        },
        {"Measure": "Selected ridge alpha", "Value": best_alpha},
        {
            "Measure": "Historical holdout starts",
            "Value": "2019-01-29",
        },
    ])

    workbook = output_folder / "Indian_Stock_Market_Analysis.xlsx"

    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        summary.to_excel(
            writer, sheet_name="Read Me Summary", index=False
        )
        market_year.to_excel(
            writer, sheet_name="Market by Year", index=False
        )
        company_summary.to_excel(
            writer, sheet_name="Company Summary", index=False
        )
        annual_returns.to_excel(
            writer, sheet_name="Company Year Returns", index=False
        )
        metrics.to_excel(
            writer, sheet_name="Forecast Accuracy", index=False
        )
        latest.to_excel(
            writer, sheet_name="Next 20 Day Forecast", index=False
        )
        test_output.to_excel(
            writer, sheet_name="Test Predictions", index=False
        )

        if skipped:
            pd.DataFrame(skipped).to_excel(
                writer, sheet_name="Skipped Files", index=False
            )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

            for column in sheet.iter_cols(
                min_row=1,
                max_row=sheet.max_row,
            ):
                heading = str(column[0].value or "").lower()

                if any(word in heading for word in [
                    "return", "drawdown", "share", "mae", "rmse", "accuracy"
                ]):
                    for cell in column[1:]:
                        cell.number_format = "0.00%"

            for cells in sheet.columns:
                letter = cells[0].column_letter
                width = min(
                    max(
                        max(
                            len(str(cell.value))
                            if cell.value is not None else 0
                            for cell in cells[:200]
                        ) + 2,
                        12,
                    ),
                    35,
                )
                sheet.column_dimensions[letter].width = width

    company_summary.to_csv(
        output_folder / "company_summary.csv", index=False
    )
    market_year.to_csv(
        output_folder / "market_by_year.csv", index=False
    )
    metrics.to_csv(
        output_folder / "forecast_accuracy.csv", index=False
    )
    latest.to_csv(
        output_folder / "next_20_day_forecast.csv", index=False
    )

    print("\nFinished.")
    print(f"Excel results: {workbook}")
    print(
        f"Company files analyzed: {len(company_summary)}; "
        f"skipped: {len(skipped)}"
    )
    print(
        f"Forecast model used {len(selected)} longest-history files; "
        f"alpha: {best_alpha}"
    )
    print("\nYearly median stock returns and market breadth:")
    print(market_year.to_string(index=False))
    print("\nForecast results (0.05 means 5%):")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()



