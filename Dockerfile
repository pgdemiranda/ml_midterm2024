# slim version of Python 3.10
FROM python:3.10.9-slim

# install dependencies necessary for building the project
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install Poetry
RUN pip install poetry

# working directory
WORKDIR /app                                                                

# copy pyproject.toml and poetry.lock
COPY ["pyproject.toml", "poetry.lock", "./"]

# to not create a virtual environment
RUN poetry config virtualenvs.create false

# install dependencies
RUN poetry install --no-root

# clean up Poetry cache to reduce image size
RUN poetry cache clear --all pypi

# copy files to the working directory
COPY ./midterm_project/predict.py ./
COPY ./midterm_project/train.py ./
COPY ./midterm_project/xgboost_model_with_preprocessor.pkl ./

# expose port 8000
EXPOSE 8000

# run the FastAPI app using Uvicorn
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]

