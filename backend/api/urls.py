"""
URL configuration for API app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("players/search/", views.player_search, name="player_search"),
    path("players/predict/", views.predict_player, name="predict_player"),
    path("players/filter/", views.filter_players, name="filter_players"),
    path("players/countries/", views.list_countries, name="list_countries"),
    path("players/highlights/", views.highlight_players, name="highlight_players"),
]
