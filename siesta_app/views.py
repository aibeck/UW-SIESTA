# Standard Library Imports
import os

# Third-Party Imports
import pandas as pd

# Django Imports
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.core.cache import caches
from django.views.decorators.csrf import csrf_exempt

# Custom Imports
from siesta_app.forms import FeatureExtractionForm, TrainingFilesForm, FileToScoreForm
from siesta_app.tasks import download_feat, train_new_model, sleep_prediction
from celery.result import AsyncResult

MEDIA = f'{settings.MEDIA_ROOT}'
CHUNK_SIZE = 1024 * 1024 * 50

# Main URLS

def index(request):
    if request.META.get("HTTP_HX_REQUEST") != 'true':
        return render(request, 'siesta_app/index_full.html')

    if not request.session.session_key:
        request.session.save()

    return render(request, "siesta_app/index.html")


def extract_features(request):
    form_data = request.session.get('form_data', None)
    form = FeatureExtractionForm(initial=form_data) if form_data else FeatureExtractionForm()

    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/extract_features.html', {'form': form})

    if not request.session.session_key:
        request.session.save()

    if request.is_ajax() and request.method == "POST":
        form = FeatureExtractionForm(request.POST, request.FILES, initial=form_data) \
            if form_data else FeatureExtractionForm(request.POST, request.FILES)

        if form.is_valid():
            form_data = form.cleaned_data

            request.session['form_data'] = form.cleaned_data
            request.session['fs'] = form_data['fs']
            request.session['ECoG_chan'] = form_data['ECoG_chan']
            request.session['EMG_chan'] = form_data['EMG_chan']
            request.session.modified = True

            extracted_data = download_feat.delay(request.session['blobName'], request.session['fs'],
                                                 request.session['ECoG_chan'] - 1, request.session['EMG_chan'] - 1)

            return JsonResponse({"task_id": extracted_data.task_id}, status=202)

        else:
            return render(request, "siesta_app/extract_features_full.html", {'form': form})
    else:
        return render(request, "siesta_app/extract_features_full.html", {'form': form})


def score_data(request):
    if request.method == 'POST':
        form = FileToScoreForm(request.POST, request.FILES)
        if form.is_valid():
            model_type = request.POST.get('model_type')

            if model_type == 'pre-trained':
                model_cache = caches['cloud'].get('trained_model')
            else:
                model_cache = os.path.join(MEDIA, request.POST.get('modelBlob'))

            csv_cache_key = 'csv_data_{}'.format(request.session.session_key)
            csv_data = form.cleaned_data['csv_file'].read()
            caches['cloud'].set(csv_cache_key, csv_data, None)

            scored_data = sleep_prediction.delay(csv_cache_key, model_cache)

            return JsonResponse({"task_id": scored_data.task_id}, status=202)
    else:
        if request.META.get("HTTP_HX_REQUEST") == 'true':
            return render(request, 'siesta_app/score_data.html', {'form': FileToScoreForm()})

    return render(request, "siesta_app/score_data_full.html", {'form': FileToScoreForm()})


def merge_data(request):
    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/merge_data.html')

    if request.method == 'POST':
        data_file = request.FILES.get('data_file')
        score_file = request.FILES.get('score_file')

        if not data_file or not score_file:
            return JsonResponse({'error': 'Both files must be provided.'}, status=400)

        try:
            merged_data = pd.read_csv(data_file, header=0, parse_dates=[-1])
            scores = pd.read_csv(score_file, header=None).iloc[:, 0].tolist()

            if len(merged_data) != len(scores):
                return JsonResponse({"error": "File lengths do not match."}, status=400)
            else: merged_data['score'] = scores

            if all(pd.to_numeric(merged_data['score'], errors='coerce').notna()):
                merged_data['score'] = merged_data['score'].map({1: 'AWAKE', 2: 'NREM', 3: 'REM'})

            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename=merged_data.csv'
            merged_data.to_csv(response, index=False)

            return response
        except Exception as e:
            error_message = 'An error occurred during when merging: {}'.format(str(e))
            return render(request, "siesta_app/merge_data_full.html", {'error_message': error_message})

    return render(request, "siesta_app/merge_data_full.html")


def fit_model(request):
    if request.META.get("HTTP_HX_REQUEST") == 'true':
        return render(request, 'siesta_app/fit_model.html', {'form': TrainingFilesForm()})

    if not request.session.session_key:
        request.session.save()

    if request.is_ajax() and request.method == "POST":
        form = TrainingFilesForm(request.POST, request.FILES)
        if form.is_valid():
            form_data = form.cleaned_data

            request.session['include_training_data'] = form_data.get('include_training_data', None)
            request.session['cache_new_model'] = form_data.get('cache_new_model', False)

            if 'blobNames' in request.session:
                new_model = train_new_model.delay(request.session['blobNames'], request.session['include_training_data'],)
            elif request.session.get('include_training_data', False):
                new_model = train_new_model.delay([], request.session['include_training_data'],)
            else:
                return JsonResponse({"error": "Invalid request"}, status=400)

            return JsonResponse({"task_id": new_model.id, "cache_new_model": request.session['cache_new_model']})

        else:
            return JsonResponse({"error": "Invalid form data"}, status=400)

    else:
        return render(request, "siesta_app/fit_model_full.html", {'form': TrainingFilesForm()})


# Helper URLS

def validate_task_state(request):
    task_id = request.GET.get('task_id')
    task_state = AsyncResult(task_id).state if task_id else "FAILURE"
    return JsonResponse({'state': task_state}, status=202)


@csrf_exempt
def upload_file(request):
    if not request.session.session_key:
        request.session.save()

    if request.method == 'POST' and request.FILES.get('file'):
        # Retrieve filename from the request
        file_name = request.POST.get('fileName')
        chunk_index = int(request.POST.get('chunkIndex'))
        total_chunks = int(request.POST.get('totalChunks'))

        file_path = os.path.join(MEDIA, file_name)

        try:
            with open(file_path, 'ab') as destination:
                for chunk in request.FILES['file'].chunks():
                    destination.write(chunk)

            if chunk_index + 1 == total_chunks:
                response_data = {'blobName': file_name}
                return JsonResponse(response_data)
            else:
                return JsonResponse({'message': 'Chunk uploaded successfully.'})
        except OSError as e:
            logger.error(f"Error writing file: {e}")
            return JsonResponse({'error': 'File upload failed.'}, status=500)

    return JsonResponse({'error': 'Invalid request.'}, status=400)


def download_csv(request):
    task_id = request.GET.get('task_id')
    filename = request.GET.get('filename')

    if task_id:
        data = pd.read_json(AsyncResult(task_id).get())

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={filename}'
        data.to_csv(response, index=False)

        return response
    else:
        return HttpResponse('No job id given.')


def download_model(request):
    task_id = request.GET.get('task_id')

    if task_id:
        model_blob = AsyncResult(task_id).get()
        cache_new_model = request.session.get('cache_new_model', False)

        file_path = os.path.join(MEDIA, model_blob)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                file_content = file.read()

            response = HttpResponse(file_content, content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename=new_model.file'
            response['X-Cache-New-Model'] = str(cache_new_model).lower()  # Convert boolean to string
            response['X-Model-Name'] = model_blob

            if not cache_new_model:
                os.remove(file_path)  # Delete the file after download

            return response

        else:
            return HttpResponse('File not found.', status=404)

    else:
        return HttpResponse('No job id given.', status=400)


def update_blob_name(request):
    if not request.session.session_key:
        request.session.save()

    request.session['blobName'] = request.POST.get('blobName') if 'blobName' in request.POST else None
    request.session['blobNames'] = request.POST.getlist('blobNames[]') if 'blobNames[]' in request.POST else None

    return JsonResponse({'status': 'success'})


def check_file_exists(request):
    model_cache = request.GET.get('modelBlob')
    if model_cache is None:
        return JsonResponse({'exists': False})

    model = os.path.join(MEDIA, model_cache)

    exists = os.path.exists(model)
    return JsonResponse({'exists': exists})