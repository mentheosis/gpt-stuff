FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN addgroup -gid 1000 gamer && adduser -uid 1000 --gid 1000 gamer
RUN mkdir /home/pygame
RUN chown -R 1000:1000 /home/pygame 

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git wget

WORKDIR /home/pygame
USER gamer

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /home/pygame/
COPY ./src /home/pygame/src

CMD ["python", "src/test-api.py"]
