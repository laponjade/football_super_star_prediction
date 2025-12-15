"""
API views for player search and prediction.
"""

import pandas as pd
from pathlib import Path
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .serializers import (
    PlayerSearchSerializer,
    PredictionRequestSerializer,
    PredictionResponseSerializer,
    FilteredPlayerSerializer,
)
from .ml_model import predict
from .feature_engineering import player_data_to_features, form_data_to_features
from .datasets import load_fifa21_dataset, build_fifa21_dataset

# Cache for player data
_player_df = None


def load_player_data():
    """Load player data from CSV file."""
    global _player_df
    if _player_df is None:
        csv_path = settings.DATA_DIR / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Player data file not found: {csv_path}")

        # Load all columns needed for the 27 features required by the model
        columns = [
            # Basic info
            "player_id",
            "short_name",
            "long_name",
            "club_name",
            "nationality_name",
            "age",
            "player_positions",
            "overall",
            "potential",
            # Version info (used to restrict to FIFA 21)
            "fifa_version",
            # Base stats
            "pace",
            "shooting",
            "passing",
            "dribbling",
            "defending",
            "physic",
            # Attacking stats
            "attacking_crossing",
            "attacking_finishing",
            "attacking_heading_accuracy",
            "attacking_short_passing",
            "attacking_volleys",
            # Skill stats
            "skill_ball_control",
            "skill_curve",
            "skill_dribbling",
            "skill_fk_accuracy",
            "skill_long_passing",
            # Movement stats
            "movement_agility",
            "movement_balance",
            "movement_reactions",
            # Power stats
            "power_jumping",
            "power_long_shots",
            "power_shot_power",
            "power_stamina",
            "power_strength",
            # Mentality stats
            "mentality_aggression",
            "mentality_composure",
            "mentality_interceptions",
            "mentality_penalties",
            "mentality_vision",
            # Defending stats
            "defending_marking_awareness",
            "defending_sliding_tackle",
            "defending_standing_tackle",
        ]

        df = pd.read_csv(csv_path, usecols=columns, low_memory=False)

        # Restrict to FIFA 21 entries only so all flows (search, predict,
        # filter, and detail pages) are consistent with the FIFA21 dataset.
        if "fifa_version" in df.columns:
            mask = df["fifa_version"].astype(float).round() == 21.0
            df = df.loc[mask].copy()

        # Drop helper column after filtering to keep original schema
        if "fifa_version" in df.columns:
            df = df.drop(columns=["fifa_version"])

        _player_df = df
        # Fill NaN values with defaults
        _player_df = _player_df.fillna(
            {
                "club_name": "Unknown",
                "nationality_name": "Unknown",
                "player_positions": "",
            }
        )

    return _player_df


@api_view(["GET"])
def filter_players(request):
    """
    Filter players (FIFA 21 subset) by country, position, and age range,
    and attach superstar prediction info for each player.

    Query parameters:
        country: nationality_name (exact match, case-insensitive)
        position: primary position token (e.g. "ST", "CM", "CB")
        min_age: minimum age (inclusive)
        max_age: maximum age (inclusive)
        limit: maximum number of results (default: 50)
    """
    country = request.query_params.get("country", "").strip()
    position = request.query_params.get("position", "").strip()
    min_age_str = request.query_params.get("min_age")
    max_age_str = request.query_params.get("max_age")
    limit = int(request.query_params.get("limit", 50))

    # Load or build FIFA 21 dataset
    fifa21_df = load_fifa21_dataset()
    if fifa21_df is None:
        fifa21_df = build_fifa21_dataset(precompute_predictions=False)

    df = fifa21_df.copy()

    # Apply country filter
    if country:
        df = df[df["nationality_name"].astype(str).str.lower().eq(country.lower())]

    # Apply position filter (match first player_positions token from original data if present;
    # otherwise assume position is encoded elsewhere and skip this filter)
    # Here we approximate by using the main position derived from other logic if available.
    if position and "player_positions" in df.columns:
        df = df[
            df["player_positions"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.upper()
            == position.upper()
        ]

    # Apply age filters if provided
    if "age" in df.columns:
        if min_age_str is not None:
            try:
                min_age = int(min_age_str)
                df = df[df["age"] >= min_age]
            except ValueError:
                pass
        if max_age_str is not None:
            try:
                max_age = int(max_age_str)
                df = df[df["age"] <= max_age]
            except ValueError:
                pass

    df = df.head(limit)

    results = []
    for _, row in df.iterrows():
        player_dict = row.to_dict()
        features = player_data_to_features(player_dict)
        prediction_result = predict(features)

        probability_pct = prediction_result["probability"] * 100
        prediction_class = int(prediction_result["prediction"])
        will_be_superstar = prediction_class == 1
        superstar_label = (
            "Will be Superstar" if will_be_superstar else "Will Not Be Superstar"
        )

        results.append(
            {
                "player_id": int(player_dict["player_id"]),
                "name": (
                    str(player_dict.get("long_name"))
                    if pd.notna(player_dict.get("long_name"))
                    else str(player_dict.get("short_name", ""))
                ),
                "club": str(player_dict.get("club_name", "Unknown")),
                "nationality": str(player_dict.get("nationality_name", "Unknown")),
                "age": int(player_dict.get("age", 20)),
                "position": str(player_dict.get("player_positions", ""))
                .split(",")[0]
                .strip(),
                "overall": int(player_dict.get("overall", 70)),
                "potential": int(player_dict.get("potential", 75)),
                "pace": (
                    int(player_dict.get("pace", 50))
                    if pd.notna(player_dict.get("pace"))
                    else 50
                ),
                "shooting": (
                    int(player_dict.get("shooting", 50))
                    if pd.notna(player_dict.get("shooting"))
                    else 50
                ),
                "passing": (
                    int(player_dict.get("passing", 50))
                    if pd.notna(player_dict.get("passing"))
                    else 50
                ),
                "dribbling": (
                    int(player_dict.get("dribbling", 50))
                    if pd.notna(player_dict.get("dribbling"))
                    else 50
                ),
                "defending": (
                    int(player_dict.get("defending", 50))
                    if pd.notna(player_dict.get("defending"))
                    else 50
                ),
                "physical": (
                    int(player_dict.get("physic", 50))
                    if pd.notna(player_dict.get("physic"))
                    else 50
                ),
                "probability": round(probability_pct, 2),
                "prediction": prediction_class,
                "will_be_superstar": will_be_superstar,
                "superstar_label": superstar_label,
            }
        )

    serializer = FilteredPlayerSerializer(results, many=True)
    return Response({"results": serializer.data}, status=status.HTTP_200_OK)


@api_view(["GET"])
def highlight_players(request):
    """
    Return curated highlight players from the FIFA 21 dataset, with
    superstar prediction info.

    - legends: top high-overall players (typically older stars)
    - prospects: top high-potential young players
    """
    # Load or build FIFA 21 dataset
    fifa21_df = load_fifa21_dataset()
    if fifa21_df is None:
        fifa21_df = build_fifa21_dataset(precompute_predictions=False)

    df = fifa21_df.copy()

    # Legends: older, very high overall
    legends_df = (
        df[(df["overall"] >= 88) & (df["age"] >= 27)]
        .sort_values(["overall", "potential"], ascending=[False, False])
        .head(3)
    )

    # Prospects: young, very high potential
    prospects_df = (
        df[(df["potential"] >= 88) & (df["age"] <= 22)]
        .sort_values(["potential", "overall"], ascending=[False, False])
        .head(10)
    )

    def build_entries(sub_df):
        entries = []
        for _, row in sub_df.iterrows():
            player_dict = row.to_dict()
            features = player_data_to_features(player_dict)
            prediction_result = predict(features)

            probability_pct = prediction_result["probability"] * 100
            prediction_class = int(prediction_result["prediction"])
            will_be_superstar = prediction_class == 1
            superstar_label = (
                "Will be Superstar" if will_be_superstar else "Will Not Be Superstar"
            )

            entries.append(
                {
                    "player_id": int(player_dict["player_id"]),
                    "name": (
                        str(player_dict.get("long_name"))
                        if pd.notna(player_dict.get("long_name"))
                        else str(player_dict.get("short_name", ""))
                    ),
                    "club": str(player_dict.get("club_name", "Unknown")),
                    "nationality": str(player_dict.get("nationality_name", "Unknown")),
                    "age": int(player_dict.get("age", 20)),
                    "position": str(player_dict.get("player_positions", ""))
                    .split(",")[0]
                    .strip(),
                    "overall": int(player_dict.get("overall", 70)),
                    "potential": int(player_dict.get("potential", 75)),
                    "pace": (
                        int(player_dict.get("pace", 50))
                        if pd.notna(player_dict.get("pace"))
                        else 50
                    ),
                    "shooting": (
                        int(player_dict.get("shooting", 50))
                        if pd.notna(player_dict.get("shooting"))
                        else 50
                    ),
                    "passing": (
                        int(player_dict.get("passing", 50))
                        if pd.notna(player_dict.get("passing"))
                        else 50
                    ),
                    "dribbling": (
                        int(player_dict.get("dribbling", 50))
                        if pd.notna(player_dict.get("dribbling"))
                        else 50
                    ),
                    "defending": (
                        int(player_dict.get("defending", 50))
                        if pd.notna(player_dict.get("defending"))
                        else 50
                    ),
                    "physical": (
                        int(player_dict.get("physic", 50))
                        if pd.notna(player_dict.get("physic"))
                        else 50
                    ),
                    "probability": round(probability_pct, 2),
                    "prediction": prediction_class,
                    "will_be_superstar": will_be_superstar,
                    "superstar_label": superstar_label,
                }
            )
        return entries

    legends = build_entries(legends_df)
    prospects = build_entries(prospects_df)

    return Response(
        {"legends": legends, "prospects": prospects}, status=status.HTTP_200_OK
    )


@api_view(["GET"])
def list_countries(request):
    """
    Return a sorted list of distinct player nationalities from the dataset.
    """
    try:
        df = load_player_data()
        countries = (
            df["nationality_name"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        return Response({"countries": countries}, status=status.HTTP_200_OK)
    except Exception:
        # If anything goes wrong, fail gracefully with an empty list
        return Response({"countries": []}, status=status.HTTP_200_OK)


@api_view(["GET"])
def player_search(request):
    """
    Search for players by name, club, or nationality.
    Also supports searching by player_id (exact match).

    Query parameters:
        q: Search query string (or player_id as string)
        limit: Maximum number of results (default: 20)
    """
    import logging

    logger = logging.getLogger(__name__)

    query = request.query_params.get("q", "").strip()
    limit = int(request.query_params.get("limit", 20))

    logger.info(f"Search request received: query='{query}', limit={limit}")

    if not query:
        return Response({"results": []}, status=status.HTTP_200_OK)

    try:
        df = load_player_data()
        logger.info(f"Loaded player data: {len(df)} players")

        # Check if query is a numeric player_id
        try:
            player_id = int(query)
            # Search by exact player_id match
            player_row = df[df["player_id"] == player_id]
            if not player_row.empty:
                results_df = player_row
            else:
                results_df = pd.DataFrame()
        except ValueError:
            # Regular text search
            query_lower = query.lower()
            logger.info(f"Searching for: '{query_lower}'")

            # Search in name, club, and nationality
            mask = (
                df["short_name"].str.lower().str.contains(query_lower, na=False)
                | df["long_name"].str.lower().str.contains(query_lower, na=False)
                | df["club_name"].str.lower().str.contains(query_lower, na=False)
                | df["nationality_name"].str.lower().str.contains(query_lower, na=False)
            )

            results_df = df[mask].head(limit)
        logger.info(f"Found {len(results_df)} matching players")

        results = []
        for _, row in results_df.iterrows():
            results.append(
                {
                    "player_id": int(row["player_id"]),
                    "name": (
                        str(row["long_name"])
                        if pd.notna(row["long_name"])
                        else str(row["short_name"])
                    ),
                    "club": (
                        str(row["club_name"])
                        if pd.notna(row["club_name"])
                        else "Unknown"
                    ),
                    "nationality": (
                        str(row["nationality_name"])
                        if pd.notna(row["nationality_name"])
                        else "Unknown"
                    ),
                    "age": int(row["age"]) if pd.notna(row["age"]) else 20,
                    "position": (
                        str(row["player_positions"]).split(",")[0].strip()
                        if pd.notna(row["player_positions"])
                        else ""
                    ),
                    "overall": int(row["overall"]) if pd.notna(row["overall"]) else 70,
                    "potential": (
                        int(row["potential"]) if pd.notna(row["potential"]) else 75
                    ),
                    "pace": int(row["pace"]) if pd.notna(row["pace"]) else 50,
                    "shooting": (
                        int(row["shooting"]) if pd.notna(row["shooting"]) else 50
                    ),
                    "passing": int(row["passing"]) if pd.notna(row["passing"]) else 50,
                    "dribbling": (
                        int(row["dribbling"]) if pd.notna(row["dribbling"]) else 50
                    ),
                    "defending": (
                        int(row["defending"]) if pd.notna(row["defending"]) else 50
                    ),
                    "physical": int(row["physic"]) if pd.notna(row["physic"]) else 50,
                }
            )

        logger.info(f"Returning {len(results)} results")
        serializer = PlayerSearchSerializer(results, many=True)
        return Response({"results": serializer.data}, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        return Response(
            {"error": f"Error searching players: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def predict_player(request):
    """
    Predict superstar potential for a player.

    Supports two modes:
    1. Predict from existing player (provide player_id)
    2. Predict from custom form input (provide player stats)
    """
    serializer = PredictionRequestSerializer(data=request.data)

    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    player_id = data.get("player_id")

    try:
        if player_id:
            # Mode 1: Predict from existing player
            df = load_player_data()
            player_row = df[df["player_id"] == player_id]

            if player_row.empty:
                return Response(
                    {"error": f"Player with ID {player_id} not found"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            player_data = player_row.iloc[0].to_dict()
            features = player_data_to_features(player_data)

        else:
            # Mode 2: Predict from form input
            form_data = {
                "age": data.get("age", 20),
                "position": data.get("position", ""),
                "pace": data.get("pace", 50),
                "shooting": data.get("shooting", 50),
                "passing": data.get("passing", 50),
                "dribbling": data.get("dribbling", 50),
                "defending": data.get("defending", 50),
                "physical": data.get("physical", 50),
                "attacking_finishing": data.get("attacking_finishing"),
                "attacking_volleys": data.get("attacking_volleys"),
                "attacking_short_passing": data.get("attacking_short_passing"),
                "attacking_heading_accuracy": data.get("attacking_heading_accuracy"),
                "skill_dribbling": data.get("skill_dribbling"),
                "skill_ball_control": data.get("skill_ball_control"),
                "mentality_interceptions": data.get("mentality_interceptions"),
                "mentality_positioning": data.get("mentality_positioning"),
                "defending_standing_tackle": data.get("defending_standing_tackle"),
                "defending_sliding_tackle": data.get("defending_sliding_tackle"),
                "power_shot_power": data.get("power_shot_power"),
                "power_long_shots": data.get("power_long_shots"),
            }
            features = form_data_to_features(form_data)

        # Make prediction
        prediction_result = predict(features)

        # Calculate probability percentage and tier
        probability_pct = prediction_result["probability"] * 100
        prediction_class = int(prediction_result["prediction"])

        # Superstar flag/label based purely on big potential class
        will_be_superstar = prediction_class == 1
        superstar_label = (
            "Will be Superstar" if will_be_superstar else "Will Not Be Superstar"
        )

        # Determine tier based on probability
        if probability_pct >= 90:
            tier = "Generational Talent 🌟"
        elif probability_pct >= 80:
            tier = "Future Superstar 🔥"
        elif probability_pct >= 70:
            tier = "Elite Potential ⚡"
        elif probability_pct >= 60:
            tier = "Strong Prospect 💪"
        elif probability_pct >= 50:
            tier = "Developing Talent 📈"
        else:
            tier = "Prospect"

        # Calculate confidence (based on how close probability is to 0 or 1)
        confidence = abs(prediction_result["probability"] - 0.5) * 2 * 100

        response_data = {
            "probability": round(probability_pct, 2),
            "prediction": prediction_class,
            "confidence": round(confidence, 2),
            "tier": tier,
            "will_be_superstar": will_be_superstar,
            "superstar_label": superstar_label,
        }

        response_serializer = PredictionResponseSerializer(response_data)
        return Response(response_serializer.data, status=status.HTTP_200_OK)

    except ValueError as e:
        return Response(
            {"error": f"Invalid input: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST
        )
    except Exception as e:
        return Response(
            {"error": f"Prediction error: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
