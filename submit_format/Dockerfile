FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

ENV SRC_DIR /src

COPY src $SRC_DIR
WORKDIR $SRC_DIR

RUN apt-get update && apt-get install -y libgtk2.0-dev
RUN python -m pip install --upgrade setuptools pip wheel
RUN pip install -r requirements.txt
RUN chmod +x ./train.sh ./inference.sh