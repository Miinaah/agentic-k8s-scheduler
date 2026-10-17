FROM python:3.13-slim

WORKDIR /app

COPY scheduler.py .

RUN pip install --no-cache-dir kubernetes==37.0.1

CMD ["python", "-u", "scheduler.py"]
