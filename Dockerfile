# Use a lightweight Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Pre-create artifacts directory (model will be mounted/copied at runtime)
RUN mkdir -p artifacts/raw artifacts/processed artifacts/models

# Expose the port that Flask will run on
EXPOSE 8080

# Command to run the app
CMD ["python", "application.py"]
