from django import forms
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator, FileExtensionValidator


class FeatureExtraction(models.Model):
    fs = models.IntegerField(verbose_name="Sampling Rate", default=400, validators=[MinValueValidator(1)])
    ECoG_chan = models.IntegerField(verbose_name="ECoG Channel", default=1, validators=[MinValueValidator(1)])
    EMG_chan = models.IntegerField(verbose_name="EMG Channel", default=3, validators=[MinValueValidator(1)])
    blob_name = models.CharField(max_length=64, db_index=True, editable=False)
    
    
class TrainingFiles(models.Model):
    include_training_data = models.BooleanField(verbose_name="Files for training", default=False)
    cache_new_model = models.BooleanField(verbose_name="Save new model to cache?", default=False)