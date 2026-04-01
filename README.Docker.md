# Docker Deployment Guide

## Quick Start

### Using Docker Compose (Recommended)

1. **Build and run the container:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f
```

3. **Stop the container:**
```bash
docker-compose down
```

### Using Docker CLI

1. **Build the image:**
```bash
docker build -t pencil2pixel-api .
```

2. **Run the container:**
```bash
docker run -d \
  --name pencil2pixel-api \
  -p 8080:8080 \
  -e SECRET_KEY="your-secret-key" \
  -e DATABASE_URL="your-database-url" \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/gfpgan/weights:/app/gfpgan/weights \
  pencil2pixel-api
```

3. **View logs:**
```bash
docker logs -f pencil2pixel-api
```

4. **Stop the container:**
```bash
docker stop pencil2pixel-api
docker rm pencil2pixel-api
```

## Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` with your values:
- `SECRET_KEY`: JWT secret key for authentication
- `DATABASE_URL`: PostgreSQL connection string

## Health Check

Check if the service is running:
```bash
curl http://localhost:8080/health
```

## API Documentation

Once running, visit:
- API Root: http://localhost:8080/
- Health Check: http://localhost:8080/health
- Interactive Docs: http://localhost:8080/docs
- OpenAPI Schema: http://localhost:8080/openapi.json

## Production Deployment

### Deploy to Cloud Platforms

#### Docker Hub
```bash
# Tag the image
docker tag pencil2pixel-api yourusername/pencil2pixel-api:latest

# Push to Docker Hub
docker push yourusername/pencil2pixel-api:latest
```

#### AWS ECR
```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag pencil2pixel-api:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/pencil2pixel-api:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/pencil2pixel-api:latest
```

#### Google Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/pencil2pixel-api

# Deploy
gcloud run deploy pencil2pixel-api \
  --image gcr.io/YOUR_PROJECT_ID/pencil2pixel-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --port 8080
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs pencil2pixel-api

# Check if port is already in use
netstat -an | grep 8080
```

### Database connection issues
- Verify DATABASE_URL is correct
- Ensure database is accessible from container
- Check firewall rules

### Out of memory
- Increase memory limit in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

## Development

### Run with hot reload
```bash
docker run -d \
  --name pencil2pixel-dev \
  -p 8080:8080 \
  -v $(pwd):/app \
  -e SECRET_KEY="dev-secret" \
  pencil2pixel-api \
  uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Access container shell
```bash
docker exec -it pencil2pixel-api bash
```
