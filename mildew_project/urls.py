"""
URL configuration for mildew_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from predictor.views import predict, predict_page 

urlpatterns = [
    path('admin/', admin.site.urls),          # 管理員頁面
    path('', predict_page, name='home'),      # 根目錄指向 predict_page
    path('predict/', predict, name='predict'), # 預測的 API 路徑
    path('predict_page/', predict_page, name='predict_page'), # 預測頁面路徑
]
