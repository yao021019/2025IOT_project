import os
import pandas as pd
import datetime 
import pickle

def prediction_proba():
    # 設置相對路徑
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 獲取當前檔案所在的目錄
    data_path = os.path.join(base_dir, "data.csv")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    model_path = os.path.join(base_dir, "final_model.pkl")
    results_path = os.path.join(base_dir, "results.csv")

    # 讀資料
    df = pd.read_csv(data_path)
    # print(df)

    # 特徵工程
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")

    target_cols = ["air_temp", "air_hum", "soil_temp", "soil_hum"]
    days = [3, 7, 14, 21]

    for col in target_cols:
        for d in days:
            df[f"{col}_mean_{d}d"] = df[col].rolling(window=d).mean()
            df[f"{col}_std_{d}d"] = df[col].rolling(window=d).std()
            df[f"{col}_sum_{d}d"] = df[col].rolling(window=d).sum()

    df["month"] = df.index.month

    def get_season(month):
        if month in [3, 4, 5]:
            return 0
        elif month in [6, 7, 8]:
            return 1
        elif month in [9, 10, 11]:
            return 2
        else:
            return 3

    df["season"] = df["month"].apply(get_season)

    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day

    df = df.reset_index(drop=False) 
    df = df.drop(columns=["date"])

    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    features = ['soil_hum_std_21d', 'soil_temp_mean_14d', 'air_temp_std_21d', 'soil_hum_std_14d', 'soil_hum_mean_7d',
                'soil_hum_sum_7d', 'soil_temp_sum_14d', 'soil_temp_std_14d', 'soil_hum_std_7d', 'day', 'soil_temp_mean_21d',
                'soil_temp_sum_21d', 'soil_temp_std_21d', 'air_temp_std_14d', 'air_hum_mean_21d', 'month',
                'air_hum_sum_21d', 'air_hum_std_21d', 'air_hum_std_3d', 'soil_temp_std_7d']

    # 只取最後一筆
    X = df[features].iloc[[-1]]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if X.isnull().any().any():
        X = X.fillna(method='ffill').fillna(method='bfill')  # 填補 NaN 值

    X_scaled = scaler.transform(X)

    if pd.isna(X_scaled).any():
        X_scaled = pd.DataFrame(X_scaled).fillna(0).values  # 用 0 填補 NaN

    y_proba = model.predict_proba(X_scaled)

    probability_of_1 = y_proba[0][1]

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')

    result = pd.DataFrame({'date': [current_date], 'probability_of_1': [probability_of_1]})

    existing_results = pd.read_csv(results_path)
    result = pd.concat([existing_results, result], ignore_index=True)

    result.to_csv(results_path, index=False)

    return probability_of_1
