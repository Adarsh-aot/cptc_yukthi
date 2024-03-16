from django.db import models

# Create your models here.
from django.db import models

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    title = models.CharField(max_length=100 , default='1970-01-01 00:00:00')
    date = models.CharField(default='1970-01-01 00:00:00' , max_length = 100)
    u_id = models.IntegerField()

class Uploaded(models.Model):
    image = models.ImageField(upload_to='upload/') 
    title = models.CharField(max_length=100 ,default='1970-01-01 00:00:00' )
    date = models.CharField(default='1970-01-01 00:00:00' , max_length = 100 )
    un = models.CharField(max_length = 100 , unique=True) 
    k_id = models.IntegerField()


from django.contrib.auth.models import User
from django.db import models

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)

    def __str__(self):
        return self.user.username