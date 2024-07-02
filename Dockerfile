FROM python:3-slim

# Update system packages and install necessary dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    expect \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Rye
RUN curl -sSL https://rye-up.com/get -o get-rye.sh && chmod +x get-rye.sh

# Expect script to automate Rye installation
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

# Run the expect script to install Rye
RUN ./install-rye.expect

# Set the working directory
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Set environment variables
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

# Expose the application port
EXPOSE 8001

# Set the entry point and command to run the application
ENTRYPOINT ["/bin/bash", "/app/rye-env.sh"]
CMD /bin/bash -c "rye sync && cd src && rye run uvicorn api.fastapi_main:app --host 0.0.0.0 --port 8001"