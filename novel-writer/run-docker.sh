docker build -t gpt-api -f ./Dockerfile .

#-p 8080:8080 \

docker run --rm --name gpt-api \
-it --entrypoint=/bin/sh \
-v "$(pwd)/src:/app/src" \
gpt-api
