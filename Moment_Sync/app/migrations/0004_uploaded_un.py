# Generated by Django 3.2.19 on 2024-03-15 15:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_auto_20240315_1925'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploaded',
            name='un',
            field=models.IntegerField(default=0, unique=True),
            preserve_default=False,
        ),
    ]
