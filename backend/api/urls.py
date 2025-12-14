"""
URL configuration for API app.
"""
from django.urls import path
from . import views

urlpatterns = [
    path('players/search/', views.player_search, name='player_search'),
    path('players/predict/', views.predict_player, name='predict_player'),
]

