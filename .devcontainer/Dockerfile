# Используем официальный образ Python
FROM mcr.microsoft.com/devcontainers/python:3.10

# Установим зависимости
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt
