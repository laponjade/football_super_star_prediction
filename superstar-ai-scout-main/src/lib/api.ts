/**
 * API client for communicating with Django backend
 */

// API base URL - can be set via VITE_API_URL environment variable
// Defaults to localhost:8000 which works for both local dev and Docker
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
  pace?: number;
  shooting?: number;
  passing?: number;
  dribbling?: number;
  defending?: number;
  physical?: number;
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

  const url = `${API_BASE_URL}/api/players/search/?q=${encodeURIComponent(query)}&limit=${limit}`;
  console.log('Search API URL:', url);

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    console.log('Search response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Search API error:', errorText);
      throw new Error(`Search failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Search API response:', data);
    return data.results || [];
  } catch (error) {
    console.error('Error searching players:', error);
    throw error;
  }
}

/**
 * Get player details by ID (searches for exact player_id match)
 */
export async function getPlayerById(playerId: number): Promise<PlayerSearchResult | null> {
  try {
    // Search for the player by searching for their ID as a string
    // This is a workaround - ideally we'd have a dedicated endpoint
    const results = await searchPlayers(playerId.toString(), 100);
    const player = results.find(p => p.player_id === playerId);
    return player || null;
  } catch (error) {
    console.error('Error getting player by ID:', error);
    return null;
  }
}

/**
 * Predict superstar potential for a player
 */
export async function predictPlayer(request: PredictionRequest): Promise<PredictionResponse> {
  const url = `${API_BASE_URL}/api/players/predict/`;
  console.log('Predict API call:', { url, request });
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    console.log('Predict API response status:', response.status);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: response.statusText }));
      console.error('Predict API error:', errorData);
      throw new Error(errorData.error || `Prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('Predict API success:', data);
    return data;
  } catch (error) {
    console.error('Error predicting player:', error);
    throw error;
  }
}

