FROM python:3.10

ENV DEBIAN_FRONTEND noninteractive
ENV TZ Asia/Tokyo

# update packages
RUN apt-get update && apt-get upgrade -y

# upgrade package manager
RUN pip install --upgrade pip

# Install pipenv
RUN pip install pipenv

# Install dependencies from requirements.txt
COPY Pipfile pyproject.toml .flake8 setup.py VERSION MANIFEST.in README.md requirements.txt ./
COPY covalent_gridengine_plugin/ ./covalent_gridengine_plugin/
RUN pipenv install --system --skip-lock --dev --verbose

# cleaning up
RUN pipenv --clear && \
    pip cache purge && \
    apt-get autoremove -y && \
    apt-get clean && rm -rf /ver/lib/apt/lists/*
