FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

RUN addgroup -gid 1000 gamer && adduser -uid 1000 --gid 1000 loureed
RUN mkdir /home/transformer
RUN chown -R 1000:1000 /home/transformer 

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y git wget

WORKDIR /home/transformer
USER loureed

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /home/transformer
COPY ./src /home/transformer/src/

CMD ["python", "src/test-api.py"]
