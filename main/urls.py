from django.contrib import admin
from django.urls import include, path
from .views import * 

app_name = 'main'

urlpatterns = [
    path('', main, name='main'),
    path('list', list, name='list'),
    path('delete', delete, name='delete'),
    path('process', process, name='process'),
    path('view', view, name='view'),
    path('logs', logs, name='logs'),
]