FROM python:3.9-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup -g 1000 myuser && adduser -D -u 1000 -G myuser myuser
USER myuser

COPY src ./src

CMD ["python", "src/test-api.py"]
