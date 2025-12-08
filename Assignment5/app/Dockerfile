# Base Python environment
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /sps_genai/helper_lib

# Copy everything into the working directory
COPY . /sps_genai/helper_lib

# Install dependencies
RUN pip install --no-cache-dir torch torchvision flask pillow

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
