docker build -t pygame -f ./Dockerfile . && \

#-p 8080:8080 \
# allow container to connect to host display per: https://medium.com/geekculture/run-a-gui-software-inside-a-docker-container-dce61771f9
docker run --rm --name pygame \
-it --entrypoint=/bin/bash \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v "$(pwd)/src:/home/pygame/src" \
-v "$(pwd)/start-game.sh:/home/pygame/start-game.sh" \
pygame
