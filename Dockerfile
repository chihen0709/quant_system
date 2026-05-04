# Use the official Playwright Python image with browser dependencies.
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

# Set the container working directory.
WORKDIR /app

# Install git for local database updates.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium.
RUN playwright install chromium

# Copy the application code into the container.
COPY . .

# Default command.
CMD ["python", "quant_pro.py"]
