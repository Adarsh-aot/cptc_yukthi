from django.contrib import admin
from .models import UploadedImage , Uploaded , UserProfile
# Register your models here.



admin.site.register(UploadedImage)

admin.site.register(Uploaded)


admin.site.register(UserProfile)