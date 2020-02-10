from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('', views.predict, name = 'predict'),
    path('record/', views.record, name='record')
]
