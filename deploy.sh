#!/bin/bash

# ==== Configuration ====
EC2_USER=ubuntu
EC2_HOST=18.234.110.70
KEY_PATH=/Users/shubhamlavaniya/Downloads/key/myec2key.pem
ECR_URL=339712962914.dkr.ecr.us-east-1.amazonaws.com/iris-model:latest
REMOTE_DIR=/home/ubuntu/iris-mlops-app

# ==== Step 1: Copy files to EC2 ====
echo "Copying project files to EC2..."
scp -i "$KEY_PATH" -r . "$EC2_USER@$EC2_HOST:$REMOTE_DIR"

# ==== Step 2: SSH and deploy using Docker Compose ====
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_HOST" << EOF
  echo "Logging into ECR..."
  aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 339712962914.dkr.ecr.us-east-1.amazonaws.com

  echo "Moving to project directory: $REMOTE_DIR"
  cd $REMOTE_DIR

  echo "Stopping and removing any existing containers..."
  docker compose down || true

  echo "Pulling latest image from ECR..."
  docker compose pull

  echo "Starting containers..."
  docker compose up -d
EOF
