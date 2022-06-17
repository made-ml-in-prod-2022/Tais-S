FROM skrivva/airflow-hw3:latest

COPY transform.py /transform.py

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

ENTRYPOINT ["python", "transform.py"]