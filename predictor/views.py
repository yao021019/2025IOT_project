from django.shortcuts import render
from .ml.prediction import prediction_proba
from django.http import JsonResponse
import pandas as pd
import os

def predict(request):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "ml", "results.csv")
    data_path = os.path.join(base_dir, "ml", "data.csv")

    probability_of_1 = prediction_proba()

    results = pd.read_csv(results_path).tail(7)
    results_json = results.to_dict(orient='records')

    # 讀取 data.csv 取得最新一筆資料
    data = pd.read_csv(data_path)
    latest_data = data.tail(1).iloc[0]  # 取最新一筆資料

    # 把需要的欄位轉換為字典
    latest_data_dict = {
        'date': latest_data['date'],
        'air_temp': latest_data['air_temp'],
        'air_hum': latest_data['air_hum'],
        'soil_temp': latest_data['soil_temp'],
        'soil_hum': latest_data['soil_hum'],
        'light_intensity': latest_data['light_intensity']
    }

    print(JsonResponse({'probability_of_1': probability_of_1, 'results': results_json}))
    # 返回預測結果、最新數據和機率
    return JsonResponse({
        'probability_of_1': probability_of_1,
        'results': results_json,
        'latest_data': latest_data_dict
    })

def predict_page(request):
    return render(request, 'predictor/predict.html')