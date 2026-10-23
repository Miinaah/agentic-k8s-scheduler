
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY split-d/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy Split C decision layer
COPY split-c/decision_layer ./decision_layer

# Copy Split D integration files
COPY split-d/scheduler_integrated.py ./scheduler_integrated.py
COPY split-d/resource_utils.py ./resource_utils.py

# Run the integrated scheduler
CMD ["python", "-u", "scheduler_integrated.py"]
