#!/bin/bash
# Open a shell in the long-lived `adexpro` container (created on first use).
# The whole repo is mounted at the same path as on the host and the shell starts
# at the repo root. GPU + X11 (RViz/Gazebo) are passed through.
# Usage: ./scripts/docker_run.sh            # shell in the container
#        ./scripts/docker_run.sh <cmd...>   # run a command in it

REPO="$(cd "$(dirname "$0")/.." && pwd)"
NAME=adexpro
IMAGE=adexpro:humble

if [ -z "$(docker images -q "$IMAGE")" ]; then
  echo "Image $IMAGE not found. Build it first (from the repo root):" >&2
  echo "  docker build -f docker/Dockerfile -t $IMAGE ." >&2
  exit 1
fi

xhost +local:docker > /dev/null 2>&1 || true

if [ -z "$(docker ps -aq -f name=^${NAME}$)" ]; then
  docker run -d --name "$NAME" \
    --gpus all --net=host --ipc=host --privileged \
    -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "$REPO":"$REPO" -w "$REPO" \
    "$IMAGE" sleep infinity > /dev/null
elif [ -z "$(docker ps -q -f name=^${NAME}$)" ]; then
  docker start "$NAME" > /dev/null
fi

if [ $# -gt 0 ]; then
  docker exec -it -w "$REPO" "$NAME" bash -ic "$*"
else
  docker exec -it -w "$REPO" "$NAME" bash
fi
