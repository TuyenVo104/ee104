FROM tensorflow/tensorflow:latest-jupyter

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

