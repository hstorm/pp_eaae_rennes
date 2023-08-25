# Use python official images. Slime or alpine is not recommended, 
# the official image already inlcudes git
# Source: https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.8

# Set the folder in which venv creates the environment
ENV WORKON_HOME /.venvs

# Install & use pipenv
# COPY Pipfile Pipfile.lock ./
COPY Pipfile  ./
RUN python -m pip install --upgrade pip
RUN pip install pipenv 
RUN pipenv install --dev

# Install application into container
# COPY . .
# WORKDIR /app
# COPY . /app
