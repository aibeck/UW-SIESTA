"""SIESTA URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf import settings
from django.conf.urls.static import static

from siesta_app import views


urlpatterns = [
    path('admin/', admin.site.urls),

    path('', views.index, name='index'),
    path('extract-features/', views.extract_features, name='extract_features'),
    path('score-data/', views.score_data, name='score_data'),
    path('merge-data/', views.merge_data, name='merge_data'),
    path('fit-model/', views.fit_model, name='fit_model'),

    path('validate-task-state/', views.validate_task_state, name='validate_task_state'),
    path('upload-file/', views.upload_file, name='upload_file'),
    path('download-csv/', views.download_csv, name='download_csv'),
    path('download-model/', views.download_model, name='download_model'),
    path('update-blob-name/', views.update_blob_name, name='update_blob_name'),
    path('check-file-exists/', views.check_file_exists, name='check_file_exists'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
