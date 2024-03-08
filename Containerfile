FROM registry.access.redhat.com/ubi9/ubi-minimal

ARG APP_ROOT=/app-root

RUN microdnf install -y --nodocs --setopt=keepcache=0 --setopt=tsflags=nodocs \
    python3.11 python3.11-devel python3.11-pip jq shadow-utils \
    && microdnf clean all --enablerepo='*'

# PYTHONDONTWRITEBYTECODE 1 : disable the generation of .pyc
# PYTHONUNBUFFERED 1 : force the stadout and stderr streams to be unbufferred
# PYTHONCOERCECLOCALE 0, PYTHONUTF8 1 : skip lgeacy locales and use UTF-8 mode
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONCOERCECLOCALE=0 \
    PYTHONUTF8=1 \
    PYTHONIOENCODING=UTF-8 \
    LANG=en_US.UTF-8 \
    PIP_NO_CACHE_DIR=off

WORKDIR ${APP_ROOT}

COPY requirements.txt ./ 
RUN pip3.11 install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt 

COPY app.py retriever.py ./ 

# Run the application
EXPOSE 8000
CMD uvicorn --host 0.0.0.0 $APP
