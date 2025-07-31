---
title: OneHealth platform deployment
hide:
- navigation
---

# Deployment of the database, API and frontend
The system is set up using [docker compose](https://docs.docker.com/compose/).

## Development environment

docker compose up --abort-on-container-exit production-runner
docker compose up api 
docker compose --profile init up --abort-on-container-exit production-runner
docker image tag onehealth-backend ghcr.io/ssciwr/onehealth-backend:latest
docker push ghcr.io/ssciwr/onehealth-backend:latest

docker volume rm $(docker volume ls -qf dangling=true)
docker volume prune
docker system prune --force --volumes

## Production environment