FROM nvcr.io/nvidia/pytorch:24.03-py3

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get -y install gcc mono-mcs g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# manually install torch dependent packages without deps
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir literate-dataclasses cebra==0.4.0 info-nce-pytorch pytorch-crf --no-deps && \
    pip install --no-cache-dir -r requirements.txt --upgrade-strategy only-if-needed
