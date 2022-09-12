FROM python:3.7-slim

ADD model_files /model_files
RUN mkdir /volume_mapping

WORKDIR /model_files
RUN pip install -r requirements.txt


CMD [ "python", "gateway_api.py" ]
