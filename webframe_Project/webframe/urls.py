from django.contrib import admin
from django.urls import path, include
from predict import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('predict/', include('predict.urls')),
]
