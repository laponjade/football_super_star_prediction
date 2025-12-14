# Django Backend for Football Superstar Prediction

This Django REST API backend serves the trained XGBoost model for predicting football player superstar potential.

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Migrations (if needed)

```bash
python manage.py migrate
```

### 3. Start the Development Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Search Players
```
GET /api/players/search/?q=<query>&limit=<number>
```
Search for players by name, club, or nationality.

**Example:**
```
GET /api/players/search/?q=mbappe&limit=10
```

**Response:**
```json
{
  "results": [
    {
      "player_id": 231747,
      "name": "Kylian Mbappé Lottin",
      "club": "Paris Saint Germain",
      "nationality": "France",
      "age": 24,
      "position": "ST",
      "overall": 91,
      "potential": 94
    }
  ]
}
```

### Predict Player Potential
```
POST /api/players/predict/
```

**Mode 1: Predict from existing player**
```json
{
  "player_id": 231747
}
```

**Mode 2: Predict from custom form input**
```json
{
  "age": 19,
  "position": "ST",
  "overall": 75,
  "potential": 85,
  "pace": 80,
  "shooting": 75,
  "passing": 70,
  "dribbling": 78,
  "defending": 40,
  "physical": 70
}
```

**Response:**
```json
{
  "probability": 75.5,
  "prediction": 1,
  "confidence": 82.3,
  "tier": "Elite Potential ⚡"
}
```

## Model Files

The backend expects the following model files in the `notebooks/` directory:
- `self_training_xgb_model.joblib` - Trained XGBoost model
- `self_training_xgb_scaler.joblib` - StandardScaler for feature normalization

## Data Files

The backend expects the following data file in the `data/` directory:
- `train.csv` - Player dataset for search functionality

## Configuration

The API base URL can be configured in the frontend via environment variable:
```
VITE_API_URL=http://localhost:8000
```

## CORS

CORS is configured to allow requests from:
- `http://localhost:5173` (Vite default)
- `http://localhost:3000`
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

To add more origins, update `CORS_ALLOWED_ORIGINS` in `football_predictor/settings.py`.

