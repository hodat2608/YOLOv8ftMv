# Generated by Django 5.1 on 2024-09-16 00:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("model_connection", "0003_model1_rotage_max_model1_rotage_min"),
    ]

    operations = [
        migrations.AddField(
            model_name="model1",
            name="dataset_format",
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
