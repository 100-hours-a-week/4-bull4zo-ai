import os
from celery import Celery

rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
rabbitmq_username = os.getenv("RABBITMQ_USERNAME", "guest")
rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")

broker_url = f"amqp://{rabbitmq_username}:{rabbitmq_password}@{rabbitmq_host}:5672//"

celery_app = Celery(
    'moa',
    broker=broker_url
)
