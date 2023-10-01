from django.urls import path, include
from . import views

urlpatterns = [
    # Define a URL pattern for the sentiment prediction view
    path('', views.predict_sentiment_view, name='predict_sentiment'),
]