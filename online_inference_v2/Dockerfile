FROM python:slim
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY models/model.pkl ./models/model.pkl
COPY app.py ./app.py

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"