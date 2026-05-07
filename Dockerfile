# Use the official Playwright Python image with browser dependencies.
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy

# Set the container working directory.
WORKDIR /app

# Set timezone without interactive prompt.
ENV TZ=Asia/Taipei
ENV DEBIAN_FRONTEND=noninteractive

# Install git and timezone data.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git tzdata && \
    ln -snf /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    echo Asia/Taipei > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages.
COPY requirements.txt requirements-dl-cpu.txt ./
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements.txt
RUN pip install --no-cache-dir --default-timeout=300 --retries 5 -r requirements-dl-cpu.txt

# Install Playwright Chromium.
RUN playwright install chromium

# Copy the application code into the container.
COPY . .

HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 CMD ["python", "healthcheck.py"]

# Default command.
CMD ["python", "quant_pro.py"]