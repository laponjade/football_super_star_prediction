# Docker Setup Guide

This guide explains how to run the Football Superstar Prediction application using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (included with Docker Desktop)

## Quick Start

### Production Build

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

3. **Stop the services:**
   ```bash
   docker-compose down
   ```

### Development Mode (with hot reload)

1. **Start services in development mode:**
   ```bash
   docker-compose -f docker-compose.dev.yml up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:5173 (Vite dev server)
   - Backend API: http://localhost:8000

## Services

### Backend (Django)
- **Port:** 8000
- **Health Check:** `/api/players/search/?q=test`
- **Volumes:**
  - `./data` → `/app/data` (read-only)
  - `./notebooks` → `/app/notebooks` (read-only)
  - `./backend` → `/app` (for development)

### Frontend (React)
- **Port:** 3000 (production) or 5173 (development)
- **Health Check:** Root endpoint
- **Environment:** `VITE_API_URL=http://localhost:8000`

## Docker Commands

### Build services
```bash
docker-compose build
```

### Start services in background
```bash
docker-compose up -d
```

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Stop services
```bash
docker-compose down
```

### Stop and remove volumes
```bash
docker-compose down -v
```

### Rebuild a specific service
```bash
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Execute commands in containers
```bash
# Backend shell
docker-compose exec backend bash

# Run Django management commands
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser

# Frontend shell
docker-compose exec frontend sh
```

## Troubleshooting

### Port already in use
If ports 8000 or 3000 are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Change 8000 to 8001
```

### Model files not found
Ensure the model files exist in the `notebooks/` directory:
- `self_training_xgb_model.joblib`
- `self_training_xgb_scaler.joblib`

### Data file not found
Ensure `train.csv` exists in the `data/` directory.

### CORS errors
The backend is configured to allow requests from `localhost`. If you're accessing from a different host, update `CORS_ALLOWED_ORIGINS` in `backend/football_predictor/settings.py`.

### Frontend can't connect to backend
1. Check that both containers are running: `docker-compose ps`
2. Check backend logs: `docker-compose logs backend`
3. Verify the API URL in frontend environment variables
4. Test backend directly: `curl http://localhost:8000/api/players/search/?q=test`

### Rebuild after code changes
```bash
# Rebuild and restart
docker-compose up --build -d

# Or rebuild specific service
docker-compose build backend
docker-compose up -d backend
```

## Production Considerations

For production deployment:

1. **Use environment variables** for sensitive settings
2. **Set `DEBUG=False`** in Django settings
3. **Use a production WSGI server** (e.g., Gunicorn) instead of Django's development server
4. **Configure proper CORS origins**
5. **Use a reverse proxy** (Nginx) for static files and SSL termination
6. **Set up proper logging**
7. **Use Docker secrets** for sensitive data

### Example Production Dockerfile for Backend

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir gunicorn
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "football_predictor.wsgi:application"]
```

## Environment Variables

You can override environment variables using a `.env` file or docker-compose environment section:

```yaml
environment:
  - DEBUG=0
  - SECRET_KEY=your-secret-key
  - VITE_API_URL=http://your-api-url
```

