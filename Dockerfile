FROM python:3.7-slim-buster
maintainer narendhrancs@gmail.com
RUN mkdir /app/
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt
COPY setup.py /app/setup.py
WORKDIR /app/
RUN pip install --upgrade pip
RUN pip install -e .
