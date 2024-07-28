from django import forms
from . import models
from django.core.validators import FileExtensionValidator

class FeatureExtractionForm(forms.ModelForm):
    class Meta:
        model = models.FeatureExtraction
        fields = '__all__'
    
class TrainingFilesForm(forms.ModelForm):
    class Meta:
        model = models.TrainingFiles
        fields = '__all__'

class FileToScoreForm(forms.Form):
    csv_file = forms.FileField(label="Choose a file to score (.csv format)", widget=forms.ClearableFileInput(attrs={'accept': '.csv'}))