# Football Superstar Prediction - Setup Guide

This project consists of a Django REST API backend and a React frontend for predicting football player superstar potential using a trained XGBoost model.

## Quick Start with Docker

The easiest way to run the application is using Docker:

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## Manual Setup

## Project Structure

```
football_super_star_prediction/
├── backend/              # Django REST API
├── data/                 # CSV datasets
├── notebooks/            # Jupyter notebooks and trained models
└── superstar-ai-scout-main/  # React frontend
```

## Prerequisites

- Python 3.8+
- Node.js 18+ (or Bun)
- pip (Python package manager)

## Backend Setup

### 1. Navigate to backend directory
```bash
cd backend
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run database migrations (optional, for admin panel)
```bash
python manage.py migrate
```

### 4. Start Django development server
```bash
python manage.py runserver
```

The backend API will be available at `http://localhost:8000`

## Frontend Setup

### 1. Navigate to frontend directory
```bash
cd superstar-ai-scout-main
```

### 2. Install dependencies
```bash
npm install
# or
bun install
```

### 3. Create environment file (optional)
Create `.env` file in the frontend directory:
```
VITE_API_URL=http://localhost:8000
```

### 4. Start development server
```bash
npm run dev
# or
bun run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in terminal)

## Usage

1. **Start the backend server** (Django) - `http://localhost:8000`
2. **Start the frontend server** (React) - `http://localhost:5173`
3. **Open the frontend** in your browser
4. Navigate to the Predict page
5. Either:
   - **Search for a player** by typing their name, club, or nationality
   - **Fill in the form** with custom player stats
6. Click "Predict Superstar DNA" to get the prediction

## Features

- **Player Search**: Search for players from the dataset with autocomplete
- **Custom Prediction**: Create a custom player profile and predict their potential
- **Real-time Predictions**: Uses trained XGBoost model via Django API
- **Visual Results**: Probability gauge, radar chart, and stat breakdowns

## Troubleshooting

### Backend Issues

- **Model not found**: Ensure `self_training_xgb_model.joblib` and `self_training_xgb_scaler.joblib` exist in `notebooks/` directory
- **Data file not found**: Ensure `train.csv` exists in `data/` directory
- **CORS errors**: Check that frontend URL is in `CORS_ALLOWED_ORIGINS` in `backend/football_predictor/settings.py`

### Frontend Issues

- **API connection failed**: Ensure backend is running and `VITE_API_URL` is correct
- **Build errors**: Try deleting `node_modules` and reinstalling dependencies

## API Documentation

See `backend/README.md` for detailed API endpoint documentation.

