/**
 * API client for communicating with Django backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PlayerSearchResult {
  player_id: number;
  name: string;
  club: string;
  nationality: string;
  age: number;
  position: string;
  overall: number;
  potential: number;
}

export interface PredictionRequest {
  // Mode 1: Predict from existing player
  player_id?: number;
  
  // Mode 2: Predict from custom form input
  age?: number;
  position?: string;
  overall?: number;
  potential?: number;
  pace?: number;
  shooting?: number;
  passing?: number;
  dribbling?: number;
  defending?: number;
  physical?: number;
  attacking_finishing?: number;
  attacking_volleys?: number;
  attacking_short_passing?: number;
  attacking_heading_accuracy?: number;
  skill_dribbling?: number;
  skill_ball_control?: number;
  mentality_interceptions?: number;
  mentality_positioning?: number;
  defending_standing_tackle?: number;
  defending_sliding_tackle?: number;
  power_shot_power?: number;
  power_long_shots?: number;
}

export interface PredictionResponse {
  probability: number;
  prediction: number;
  confidence: number;
  tier: string;
}

/**
 * Search for players by name, club, or nationality
 */
export async function searchPlayers(query: string, limit: number = 20): Promise<PlayerSearchResult[]> {
  if (!query.trim()) {
    return [];
  }

  try {
    const response = await fetch(
      `${API_BASE_URL}/api/players/search/?q=${encodeURIComponent(query)}&limit=${limit}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.results || [];
  } catch (error) {
    console.error('Error searching players:', error);
    throw error;
  }
}

/**
 * Predict superstar potential for a player
 */
export async function predictPlayer(request: PredictionRequest): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/players/predict/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorData.error || `Prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error predicting player:', error);
    throw error;
  }
}

