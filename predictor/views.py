from django.shortcuts import render
from .ml.prediction import prediction_proba
from django.http import JsonResponse
import pandas as pd
import os

def predict(request):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(base_dir, "ml", "results.csv")

    probability_of_1 = prediction_proba()

    results = pd.read_csv(results_path).tail(7)
    results_json = results.to_dict(orient='records')

    print(JsonResponse({'probability_of_1': probability_of_1, 'results': results_json}))
    return JsonResponse({'probability_of_1': probability_of_1, 'results': results_json})

def predict_page(request):
    return render(request, 'predictor/predict.html')