FROM python:3.7-slim-buster
maintainer narendhrancs@gmail.com
RUN mkdir /app/
RUN pip install --upgrade pip
RUN pip install --upgrade othoz-adding-sum
COPY tutorial/ /app/tutorial/
COPY train_inference.sh /app/train_inference.sh
WORKDIR /app/
CMD sh /app/train_inference.sh