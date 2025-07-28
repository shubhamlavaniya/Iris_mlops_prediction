#!/bin/bash

EC2_USER=ubuntu
EC2_HOST=52.90.150.186
KEY_PATH=/Users/shubhamlavaniya/Downloads/keys/myec2key.pem
ECR_URL=339712962914.dkr.ecr.us-east-1.amazonaws.com/iris-model:latest

ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST << EOF
  echo "Logging into ECR..."
  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 339712962914.dkr.ecr.us-east-1.amazonaws.com
  echo "Pulling latest Docker image..."
  docker pull $ECR_URL
  echo "Stopping old container if running..."
  docker stop iris-app || true && docker rm iris-app || true
  echo "Running new container..."
  docker run -d --name iris-app -p 8000:8000 $ECR_URL
EOF
