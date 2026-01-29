# Football Superstar Prediction

A full-stack web application for predicting football player superstar potential using machine learning (XGBoost).

## Features

- 🔍 **Player Search**: Search for players from the dataset with autocomplete
- 📊 **Custom Prediction**: Create a custom player profile and predict their potential
- 🤖 **ML-Powered**: Uses trained XGBoost model for accurate predictions
- 🎨 **Modern UI**: Beautiful React frontend with real-time predictions
- 🐳 **Docker Support**: Easy deployment with Docker Compose

## Quick Start with Docker

The easiest way to run the application:

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
```

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

## Manual Setup

See [SETUP.md](SETUP.md) for manual installation instructions.

## Project Structure

```
football_super_star_prediction/
├── backend/              # Django REST API
│   ├── api/             # API endpoints and ML integration
│   └── football_predictor/  # Django project settings
├── data/                 # CSV datasets
├── notebooks/            # Jupyter notebooks and MLOps pipeline
│   ├── mlops_*.py       # MLOps scripts (data versioning, training, sweeps, registry)
│   ├── run_mlops_pipeline.py  # Main pipeline runner
│   └── MLOPS_README.md  # MLOps documentation
└── superstar-ai-scout-main/  # React frontend
```

## MLOps Pipeline

This project includes a complete MLOps implementation using Weights & Biases (W&B):

- ✅ **Data Versioning** - Datasets versioned as W&B artifacts
- ✅ **Experiment Tracking** - All training runs tracked with metrics
- ✅ **Hyperparameter Optimization** - Bayesian sweeps for model tuning
- ✅ **Model Registration** - Best models registered as versioned artifacts

### Quick Start (MLOps)

```bash
cd notebooks
python run_mlops_pipeline.py
# Choose option 5: All phases (1, 2, 3, 4)
```

For detailed MLOps documentation, see [notebooks/MLOPS_README.md](notebooks/MLOPS_README.md).

## API Endpoints

- `GET /api/players/search/?q=<query>` - Search players
- `POST /api/players/predict/` - Predict player potential

See [backend/README.md](backend/README.md) for detailed API documentation.

## Technologies

- **Backend**: Django, Django REST Framework, XGBoost, scikit-learn
- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **ML**: XGBoost, scikit-learn, joblib
- **Deployment**: Docker, Docker Compose

## License

This project is for educational purposes.
