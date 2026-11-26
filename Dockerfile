# Required Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Disable Python cache and enable immediate logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies before copying source code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application and trained model
COPY app/ ./app/
COPY models/ ./models/

# Copy configuration and source modules
COPY params.yaml .
COPY src/ ./src/

# Expose the API port
EXPOSE 8000

# Start FastAPI inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]