"""
ASGI config for Titanic_Survial_Prediction_Web project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Titanic_Survial_Prediction_Web.settings')

application = get_asgi_application()
