FROM python:3.10-slim-buster

RUN pip install explainerdashboard pandas scikit-learn

COPY dashboard.py ./
COPY app.py ./
COPY dataset.csv ./

RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]