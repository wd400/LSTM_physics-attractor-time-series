FROM python:3.12-slim

WORKDIR /app

# Install git and dvc dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

ENTRYPOINT ["python", "run.py"]