release: python manage.py collectstatic --noinput 
web: gunicorn SIESTA.wsgi:application
#web: waitress-serve SIESTA.wsgi:application --timeout 120 --log-file -
#worker: celery -A SIESTA worker -l info -c 4