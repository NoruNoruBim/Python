from django.contrib import admin
from django.urls import re_path
from django.urls import path
from firstapp import views

urlpatterns = [
    path("admin", views.admin),
    re_path(r"^", views.index),
]
