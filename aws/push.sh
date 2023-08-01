sh compile.sh

docker build -t azcausal-lambda-run .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 112353327285.dkr.ecr.us-east-1.amazonaws.com
docker tag azcausal-lambda-run:latest 112353327285.dkr.ecr.us-east-1.amazonaws.com/azcausal-lambda-run:latest
docker push 112353327285.dkr.ecr.us-east-1.amazonaws.com/azcausal-lambda-run:latest
