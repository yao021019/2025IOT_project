from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('predict_page/', views.predict_page, name='predict_page'),
]
