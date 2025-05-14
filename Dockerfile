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

    # --- Docker Build Debugging Steps ---
    RUN echo "--- Docker Build Log: Verifying contents of /app ---" && ls -la /app
    RUN echo "--- Docker Build Log: Contents of /app/set_env.py ---" && cat /app/set_env.py
    RUN echo "--- Docker Build Log: Attempting to execute /app/set_env.py directly ---"
    # Run set_env.py and allow it to fail without stopping the build, but print its output
    RUN python /app/set_env.py || (echo "--- Docker Build Log: set_env.py execution failed or had non-zero exit ---" && exit 0)
    RUN echo "--- Docker Build Log: Finished attempting to execute /app/set_env.py ---"
    # --- End Docker Build Debugging Steps ---

    # Make port available (Gunicorn will bind to $PORT)
    ENV PORT 8000
    EXPOSE 8000 # This just documents the port, Gunicorn uses $PORT

    # Command to run the application using Gunicorn
    CMD sh -c 'gunicorn --bind "0.0.0.0:$PORT" app:app'
    