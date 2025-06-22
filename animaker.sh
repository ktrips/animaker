#!/bin/bash
set -euo pipefail

echo "Starting Docker BUILD"
docker build --platform=linux/amd64 -t asia-northeast1-docker.pkg.dev/raspberryai/animaker-repo/animaker:v1 .

#echo "Local/mac Docker BUILD"
#docker build --platform=linux/arm64/v8 -t asia-northeast1-docker.pkg.dev/raspberryai/animaker-repo/animaker:v1 .
#echo "Local Docker Run -8080"
#docker run -p 8080:8080 asia-northeast1-docker.pkg.dev/raspberryai/animaker-repo/animaker:v1

echo "Starting Docker PUSH"
docker push asia-northeast1-docker.pkg.dev/raspberryai/animaker-repo/animaker:v1

echo "Gcloud run deploy"
gcloud run deploy animaker --image=asia-northeast1-docker.pkg.dev/raspberryai/animaker-repo/animaker:v1 --memory=1Gi --port=8080 --allow-unauthenticated --region=asia-northeast1

echo "Gcloud deploy completed"