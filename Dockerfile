FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update && apt-get install -y texlive-xetex texlive-fonts-recommended texlive-plain-generic
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install -U ipykernel

