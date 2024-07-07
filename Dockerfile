# Use the official Python image as a base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /footfallv8

# Install system dependencies including libGL.so.1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1-mesa-glx \
       build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt update \
    && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install Python dependencies
COPY requirements.txt /footfallv8/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Django project
COPY . /footfallv8/

# Copy the init script and make it executable
COPY init.sh /footfallv8/init.sh
RUN chmod +x /footfallv8/init.sh

# Run the init script
CMD ["./init.sh"]
