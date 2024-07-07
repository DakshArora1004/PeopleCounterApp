#!/bin/bash

# Apply database migrations
echo "Making database migrations..."
python manage.py makemigrations
echo "Applying database migrations..."
python manage.py migrate

# Collect static files (only if you are serving static files with Django)
# echo "Collecting static files..."
# python manage.py collectstatic --noinput

# Create a superuser if one does not exist
# SUPERUSER_EMAIL=${DJANGO_SUPERUSER_EMAIL:-"admin@example.com"}
# SUPERUSER_PASSWORD=${DJANGO_SUPERUSER_PASSWORD:-"admin"}
# DJANGO_SUPERUSER_USERNAME=${DJANGO_SUPERUSER_USERNAME:-"admin"}

# echo "Creating superuser..."
# echo "from django.contrib.auth import get_user_model; User = get_user_model(); \
# if not User.objects.filter(username='${DJANGO_SUPERUSER_USERNAME}').exists(): \
#     User.objects.create_superuser('${DJANGO_SUPERUSER_USERNAME}', '${SUPERUSER_EMAIL}', '${SUPERUSER_PASSWORD}')" | python manage.py shell

# Start the Django development server
echo "Starting Django development server..."
exec python manage.py runserver 0.0.0.0:8000
