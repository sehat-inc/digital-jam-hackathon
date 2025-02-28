# Use an official Python image
FROM python:3.10

# Set the working directory
WORKDIR /api-flask

# Copy requirements file
COPY requirements.txt /api-flask/requirements.txt

# Install system dependencies (including Rust and Cargo)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    libssl-dev \
    libffi-dev \
    rustc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r /api-flask/requirements.txt

# Copy project files
COPY . /api-flask/

# Expose port for Flask
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
