FROM python:3.12-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the entrypoint to forward arguments to python cli.py
ENTRYPOINT ["python", "cli.py"]

