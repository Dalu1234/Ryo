# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 80 available to the world outside this container (Render will map this)
# Gunicorn will bind to the port specified by Render's PORT env var,
# but exposing a default like 80 or a common one is good practice.
# Render will likely use its own internal port mapping.
EXPOSE 8000

# Define environment variable for the PORT Gunicorn should listen on
# Render will set this, but having a default can be useful.
ENV PORT 8000

# Command to run the application using Gunicorn
# Render will inject the actual $PORT it wants Gunicorn to listen on.
# The 'app:app' refers to the 'app' Flask instance in your 'app.py' file.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]