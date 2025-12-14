"""
Feature engineering to convert player data to model input features.
"""
import pandas as pd
import numpy as np


# Required feature order for the model (27 features as per saved model)
REQUIRED_FEATURES = [
    'age',
    'physic',
    'mentality_aggression',
    'mentality_interceptions',
    'power_stamina',
    'power_strength',
    'defending_marking_awareness',
    'power_jumping',
    'defending_standing_tackle',
    'defending_sliding_tackle',
    'attacking_heading_accuracy',
    'mentality_composure',
    'movement_reactions',
    'skill_long_passing',
    'skill_dribbling',
    'skill_fk_accuracy',
    'skill_ball_control',
    'attacking_crossing',
    'power_shot_power',
    'attacking_finishing',
    'skill_curve',
    'movement_balance',
    'attacking_volleys',
    'power_long_shots',
    'mentality_vision',
    'mentality_penalties',
    'movement_agility'
]


def extract_position_flags(position):
    """
    Extract position flags from position string.
    
    Args:
        position: String like "ST", "LW", "CM", "CB", etc.
    
    Returns:
        tuple: (attacker_position, midfielder_position)
    """
    position_upper = str(position).upper() if position else ""
    
    # Attacker positions
    attacker_positions = ['ST', 'CF', 'LW', 'RW', 'LF', 'RF', 'LS', 'RS']
    # Midfielder positions
    midfielder_positions = ['CAM', 'CM', 'CDM', 'LM', 'RM', 'LAM', 'RAM', 'LCM', 'RCM', 'LDM', 'RDM']
    
    attacker_position = 1 if any(pos in position_upper for pos in attacker_positions) else 0
    midfielder_position = 1 if any(pos in position_upper for pos in midfielder_positions) else 0
    
    return attacker_position, midfielder_position


def calculate_shooting(attacking_finishing, power_shot_power, power_long_shots):
    """
    Calculate shooting feature (average of finishing, shot power, and long shots).
    
    Args:
        attacking_finishing: Finishing stat
        power_shot_power: Shot power stat
        power_long_shots: Long shots stat
    
    Returns:
        float: Average shooting stat
    """
    if attacking_finishing is None or power_shot_power is None or power_long_shots is None:
        return 50.0  # Default value
    
    return (attacking_finishing + power_shot_power + power_long_shots) / 3.0


def player_data_to_features(player_data):
    """
    Convert player data (from CSV or form) to model features.
    
    Args:
        player_data: dict or pandas Series with player information
    
    Returns:
        numpy array: Array of 27 features in the correct order
    """
    # Handle both dict and Series
    if isinstance(player_data, pd.Series):
        data = player_data.to_dict()
    else:
        data = player_data
    
    # Helper function to safely get float value
    def safe_float(value, default=50.0):
        if value is None or pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Build features array in the exact order required (27 features)
    features = [
        safe_float(data.get('age'), 20),
        safe_float(data.get('physic'), 50),
        safe_float(data.get('mentality_aggression'), 50),
        safe_float(data.get('mentality_interceptions'), 50),
        safe_float(data.get('power_stamina'), 50),
        safe_float(data.get('power_strength'), 50),
        safe_float(data.get('defending_marking_awareness'), 50),
        safe_float(data.get('power_jumping'), 50),
        safe_float(data.get('defending_standing_tackle'), 50),
        safe_float(data.get('defending_sliding_tackle'), 50),
        safe_float(data.get('attacking_heading_accuracy'), 50),
        safe_float(data.get('mentality_composure'), 50),
        safe_float(data.get('movement_reactions'), 50),
        safe_float(data.get('skill_long_passing'), 50),
        safe_float(data.get('skill_dribbling'), 50),
        safe_float(data.get('skill_fk_accuracy'), 50),
        safe_float(data.get('skill_ball_control'), 50),
        safe_float(data.get('attacking_crossing'), 50),
        safe_float(data.get('power_shot_power'), 50),
        safe_float(data.get('attacking_finishing'), 50),
        safe_float(data.get('skill_curve'), 50),
        safe_float(data.get('movement_balance'), 50),
        safe_float(data.get('attacking_volleys'), 50),
        safe_float(data.get('power_long_shots'), 50),
        safe_float(data.get('mentality_vision'), 50),
        safe_float(data.get('mentality_penalties'), 50),
        safe_float(data.get('movement_agility'), 50),
    ]
    
    return np.array(features)


def form_data_to_features(form_data):
    """
    Convert form input data to model features.
    
    Args:
        form_data: dict with form fields:
            - age
            - position
            - pace, shooting, passing, dribbling, defending, physical
            - (optional: other detailed stats)
    
    Returns:
        numpy array: Array of 27 features in the correct order
    """
    # Helper function to safely get float value, handling None
    def safe_float(value, default=50.0):
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Get base stats with defaults
    age = safe_float(form_data.get('age'), 20.0)
    physical = safe_float(form_data.get('physical'), 50.0)
    passing = safe_float(form_data.get('passing'), 50.0)
    dribbling = safe_float(form_data.get('dribbling'), 50.0)
    defending = safe_float(form_data.get('defending'), 50.0)
    shooting = safe_float(form_data.get('shooting'), 50.0)
    
    # Get detailed stats, using base stats as fallback where appropriate
    attacking_finishing = safe_float(form_data.get('attacking_finishing'), shooting)
    attacking_volleys = safe_float(form_data.get('attacking_volleys'), shooting)
    attacking_crossing = safe_float(form_data.get('attacking_crossing'), passing)
    attacking_heading_accuracy = safe_float(form_data.get('attacking_heading_accuracy'), 50.0)
    skill_dribbling = safe_float(form_data.get('skill_dribbling'), dribbling)
    skill_ball_control = safe_float(form_data.get('skill_ball_control'), dribbling)
    skill_long_passing = safe_float(form_data.get('skill_long_passing'), passing)
    skill_fk_accuracy = safe_float(form_data.get('skill_fk_accuracy'), 50.0)
    skill_curve = safe_float(form_data.get('skill_curve'), 50.0)
    power_shot_power = safe_float(form_data.get('power_shot_power'), attacking_finishing)
    power_long_shots = safe_float(form_data.get('power_long_shots'), attacking_finishing)
    power_stamina = safe_float(form_data.get('power_stamina'), physical)
    power_strength = safe_float(form_data.get('power_strength'), physical)
    power_jumping = safe_float(form_data.get('power_jumping'), physical)
    defending_standing_tackle = safe_float(form_data.get('defending_standing_tackle'), defending)
    defending_sliding_tackle = safe_float(form_data.get('defending_sliding_tackle'), defending)
    defending_marking_awareness = safe_float(form_data.get('defending_marking_awareness'), defending)
    mentality_interceptions = safe_float(form_data.get('mentality_interceptions'), 50.0)
    mentality_composure = safe_float(form_data.get('mentality_composure'), 50.0)
    mentality_aggression = safe_float(form_data.get('mentality_aggression'), 50.0)
    mentality_vision = safe_float(form_data.get('mentality_vision'), passing)
    mentality_penalties = safe_float(form_data.get('mentality_penalties'), 50.0)
    movement_reactions = safe_float(form_data.get('movement_reactions'), 50.0)
    movement_balance = safe_float(form_data.get('movement_balance'), 50.0)
    movement_agility = safe_float(form_data.get('movement_agility'), 50.0)
    
    # Build features array in the exact order required (27 features)
    features = [
        age,
        physical,
        mentality_aggression,
        mentality_interceptions,
        power_stamina,
        power_strength,
        defending_marking_awareness,
        power_jumping,
        defending_standing_tackle,
        defending_sliding_tackle,
        attacking_heading_accuracy,
        mentality_composure,
        movement_reactions,
        skill_long_passing,
        skill_dribbling,
        skill_fk_accuracy,
        skill_ball_control,
        attacking_crossing,
        power_shot_power,
        attacking_finishing,
        skill_curve,
        movement_balance,
        attacking_volleys,
        power_long_shots,
        mentality_vision,
        mentality_penalties,
        movement_agility,
    ]
    
    return np.array(features)

