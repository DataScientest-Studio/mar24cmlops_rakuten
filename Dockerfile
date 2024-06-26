FROM python:3-slim

# Mettre à jour les packages système et installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    expect \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Télécharger et installer Rye
RUN curl -sSL https://rye-up.com/get -o get-rye.sh && chmod +x get-rye.sh

# Script expect pour automatiser l'installation de Rye
RUN echo '#!/usr/bin/expect -f\n\
set timeout -1\n\
spawn ./get-rye.sh\n\
expect "? Continue? (y/n)" { send "y\r" }\n\
expect "Select the preferred package installer" { send "\r" }\n\
expect "What should running `python` or `python3` do when you are not inside a Rye managed project?" { send "\r" }\n\
expect "Which version of Python should be used as default toolchain? (cpython@3.12)" { send "3.10.11\r" }\n\
expect "Should the installer add Rye to PATH via .profile?" { send "y\r" }\n\
expect eof' > install-rye.expect \
    && chmod +x install-rye.expect

# Exécuter le script expect pour installer Rye
RUN ./install-rye.expect

WORKDIR /app

COPY . .

ENV PORT=8001
ENV AWS_CONFIG_PATH=/app/.aws/.encrypted
ENV AWS_CONFIG_FOLDER=/app/.aws/
ENV DATA_PATH=/app/data/
ENV S3_INIT_DB_PATH=rakuten_init.duckdb
ENV RAKUTEN_DB_NAME=rakuten_db.duckdb
ENV S3_BUCKET=rakutenprojectbucket
ENV KEY=ZDDVMAi_s9Jn03haGGSzoDVxPtd99XM2593PyNsyBbc=
ENV JWT_KEY=0e3e3c66376ed252999eaf1abe176789552e7d69ac20d25824989fa6dd362093
ENV ACCESS_TOKEN_EXPIRE_MINUTES=30
ENV ALGORITHM=HS256
EXPOSE 8001

ENTRYPOINT ["/bin/bash", "/app/rye-env.sh"]
CMD /bin/bash -c "rye sync && rye run python ./src/api/fastapi_main.py"