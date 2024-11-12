# Slim version of Python 3.10
FROM python:3.10.9-slim

# Install dependencies necessary for building the project
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install specific version of Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app                                                                

# Copy the necessary files to the working directory (pyproject.toml and poetry.lock)
COPY ["pyproject.toml", "poetry.lock", "./"]

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-root

# Clean up Poetry cache to reduce image size
RUN poetry cache clear --all pypi

# Copy files to the working directory
COPY ./midterm_project/predict.py ./
COPY ./midterm_project/train.py ./
COPY ./midterm_project/xgboost_model_with_preprocessor.pkl ./


# Expose the correct port (matching Uvicorn's host and port setting)
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]

