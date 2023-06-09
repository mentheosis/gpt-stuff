#docker build -t transformer -f ./Dockerfile . && \

docker run --rm --gpus all --name transformer \
-it --entrypoint=/bin/bash \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
-v ~/Downloads/scripts_ds9:/home/transformer/dataset \
-v "$(pwd)/src:/home/transformer/src" \
transformer

# allow container to connect to host display per: https://medium.com/geekculture/run-a-gui-software-inside-a-docker-container-dce61771f9
#-e DISPLAY=$DISPLAY \
#-p 8080:8080 \
