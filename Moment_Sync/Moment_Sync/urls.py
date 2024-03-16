"""Moment_Sync URL Configuration

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
from django.urls import path
from django.contrib import admin
from app import views
from django.conf import settings

from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', views.upload_image, name='upload_image'),
    path('success/', views.upload_success, name='upload_success'),
    path('check_image/', views.check_image, name='check_image'),
    path('myimage/<int:id>/', views.myimage, name='myimage'),
    path('verified_image/<int:id>/', views.verified_image, name='verified_image'),
    path('delete_pic/<int:id>/', views.delete_pic, name='delete_pic'),
    path('delete_picc/<int:id>/', views.delete_picc, name='delete_picc'),
    path('delete_p/<int:id>/', views.delete_p, name='delete_p'),
    path('', views.register_view, name='register'),
    path('upload-image/', views.upload_image, name='upload_image'),
    path('home/', views.home, name='home'),
    path('camera/', views.camera, name='camera'),
    path('upload_phot/', views.upload_p, name='upload_p'),
    path('login/', views.loginn, name='login'),
    path('past_image/', views.past_images, name='past_images'),
    path('apply_filter/', views.apply_filter, name='apply_filter'),
    path('f_man/', views.f_man, name='f_man'),\
    path('uploadfile/', views.upload_file, name='upload_file'),
    path('downloadfile/', views.download_file, name='download_file'),
    path('query/', views.query, name='query'),
    # path('api_request/', views.api_request, name='api_request'),
    
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)