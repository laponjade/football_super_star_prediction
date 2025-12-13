"""
Serializers for API responses.
"""
from rest_framework import serializers


class PlayerSearchSerializer(serializers.Serializer):
    """Serializer for player search results."""
    player_id = serializers.IntegerField()
    name = serializers.CharField()
    club = serializers.CharField(allow_null=True)
    nationality = serializers.CharField(allow_null=True)
    age = serializers.IntegerField()
    position = serializers.CharField()
    overall = serializers.IntegerField()
    potential = serializers.IntegerField()
    pace = serializers.IntegerField(required=False)
    shooting = serializers.IntegerField(required=False)
    passing = serializers.IntegerField(required=False)
    dribbling = serializers.IntegerField(required=False)
    defending = serializers.IntegerField(required=False)
    physical = serializers.IntegerField(required=False)


class PredictionRequestSerializer(serializers.Serializer):
    """Serializer for prediction requests."""
    # Mode 1: Predict from existing player
    player_id = serializers.IntegerField(required=False, allow_null=True)
    
    # Mode 2: Predict from custom form input
    age = serializers.IntegerField(required=False, min_value=16, max_value=50)
    position = serializers.CharField(required=False, allow_blank=True)
    overall = serializers.IntegerField(required=False, min_value=40, max_value=99)
    potential = serializers.IntegerField(required=False, min_value=40, max_value=99)
    pace = serializers.IntegerField(required=False, min_value=1, max_value=99)
    shooting = serializers.IntegerField(required=False, min_value=1, max_value=99)
    passing = serializers.IntegerField(required=False, min_value=1, max_value=99)
    dribbling = serializers.IntegerField(required=False, min_value=1, max_value=99)
    defending = serializers.IntegerField(required=False, min_value=1, max_value=99)
    physical = serializers.IntegerField(required=False, min_value=1, max_value=99)
    
    # Optional detailed stats
    attacking_finishing = serializers.IntegerField(required=False, allow_null=True)
    attacking_volleys = serializers.IntegerField(required=False, allow_null=True)
    attacking_short_passing = serializers.IntegerField(required=False, allow_null=True)
    attacking_heading_accuracy = serializers.IntegerField(required=False, allow_null=True)
    skill_dribbling = serializers.IntegerField(required=False, allow_null=True)
    skill_ball_control = serializers.IntegerField(required=False, allow_null=True)
    mentality_interceptions = serializers.IntegerField(required=False, allow_null=True)
    mentality_positioning = serializers.IntegerField(required=False, allow_null=True)
    defending_standing_tackle = serializers.IntegerField(required=False, allow_null=True)
    defending_sliding_tackle = serializers.IntegerField(required=False, allow_null=True)
    power_shot_power = serializers.IntegerField(required=False, allow_null=True)
    power_long_shots = serializers.IntegerField(required=False, allow_null=True)
    
    def validate(self, data):
        """Validate that either player_id or form data is provided."""
        player_id = data.get('player_id')
        has_form_data = any([
            data.get('age'),
            data.get('position'),
            data.get('overall'),
            data.get('potential'),
        ])
        
        if not player_id and not has_form_data:
            raise serializers.ValidationError(
                "Either 'player_id' or form data (age, position, overall, potential) must be provided."
            )
        
        if player_id and has_form_data:
            raise serializers.ValidationError(
                "Cannot provide both 'player_id' and form data. Use one or the other."
            )
        
        return data


class PredictionResponseSerializer(serializers.Serializer):
    """Serializer for prediction responses."""
    probability = serializers.FloatField()
    prediction = serializers.IntegerField()
    confidence = serializers.FloatField()
    tier = serializers.CharField()

