#!/bin/bash

docker login -u clvgt12 -p $(pass show docker-credential-helpers/aHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEv/clvgt12)
for img in api pdf_bot loader bot front-end
do
    docker tag genai-stack-$img clvgt12/genai-stack-$img:amd64_latest
    docker push clvgt12/genai-stack-$img:amd64_latest
done
docker logout
