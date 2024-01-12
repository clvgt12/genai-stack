#!/bin/bash

docker login -u clvgt12 -p $(pass show Docker/docker_hub_access_token)
for img in api pdf_bot loader bot front-end
do
    docker tag genai-stack-$img clvgt12/genai-stack-$img:amd64_latest
    docker push clvgt12/genai-stack-$img:amd64_latest
done
docker logout
