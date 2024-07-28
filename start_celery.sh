#!/bin/bash

source /home/siesta/.venv/bin/activate

cd /home/siesta/UW-SIESTA

celery -A SIESTA.celery worker -l INFO -c 8 --heartbeat-interval=300
