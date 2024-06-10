FROM python:3-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8001
ENV AWS_CONFIG_PATH=/app/.aws/.encrypted
ENV AWS_CONFIG_FOLDER=/app/.aws/
ENV DATA_PATH=/app/data/
ENV S3_INIT_DB_PATH=rakuten_init.duckdb
ENV RAKUTEN_DB_NAME=rakuten_db.duckdb
ENV S3_BUCKET=rakutenprojectbucket
ENV KEY=2GG5s08SC49gwj8vBe0Ni-mDJP907MPMt012MX1Tkkg=

EXPOSE 8001

CMD [ "python", "./src/api/fastapi_main.py" ]