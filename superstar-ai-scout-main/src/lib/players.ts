export interface Player {
  id: number;
  name: string;
  age: number;
  position: string;
  nationality: string;
  club: string;
  overall: number;
  potential: number;
  pace: number;
  shooting: number;
  passing: number;
  dribbling: number;
  defending: number;
  physical: number;
  image?: string;
}

export interface PredictionResult {
  probability: number;
  confidence: number;
  tier: string;
  comparisons: {
    stat: string;
    label: string;
    emoji: string;
    comparison: string;
  }[];
  benchmarkStats: number[];
  userStats: number[];
}

// Mock U21 players data
export const mockPlayers: Player[] = [
  { id: 1, name: "Kylian Mbappé", age: 25, position: "ST", nationality: "France", club: "Real Madrid", overall: 91, potential: 95, pace: 97, shooting: 89, passing: 80, dribbling: 92, defending: 36, physical: 78 },
  { id: 2, name: "Erling Haaland", age: 24, position: "ST", nationality: "Norway", club: "Man City", overall: 91, potential: 94, pace: 89, shooting: 94, passing: 66, dribbling: 80, defending: 45, physical: 88 },
  { id: 3, name: "Jude Bellingham", age: 21, position: "CM", nationality: "England", club: "Real Madrid", overall: 90, potential: 94, pace: 76, shooting: 82, passing: 84, dribbling: 87, defending: 72, physical: 80 },
  { id: 4, name: "Florian Wirtz", age: 21, position: "CAM", nationality: "Germany", club: "Leverkusen", overall: 88, potential: 93, pace: 78, shooting: 82, passing: 86, dribbling: 89, defending: 42, physical: 62 },
  { id: 5, name: "Lamine Yamal", age: 17, position: "RW", nationality: "Spain", club: "Barcelona", overall: 81, potential: 95, pace: 91, shooting: 70, passing: 79, dribbling: 88, defending: 28, physical: 50 },
  { id: 6, name: "Jamal Musiala", age: 21, position: "CAM", nationality: "Germany", club: "Bayern", overall: 87, potential: 93, pace: 79, shooting: 78, passing: 83, dribbling: 91, defending: 40, physical: 64 },
  { id: 7, name: "Pedri", age: 21, position: "CM", nationality: "Spain", club: "Barcelona", overall: 87, potential: 92, pace: 71, shooting: 70, passing: 87, dribbling: 89, defending: 68, physical: 63 },
  { id: 8, name: "Gavi", age: 20, position: "CM", nationality: "Spain", club: "Barcelona", overall: 83, potential: 90, pace: 72, shooting: 68, passing: 81, dribbling: 83, defending: 72, physical: 75 },
  { id: 9, name: "Bukayo Saka", age: 22, position: "RW", nationality: "England", club: "Arsenal", overall: 86, potential: 91, pace: 84, shooting: 79, passing: 83, dribbling: 87, defending: 65, physical: 68 },
  { id: 10, name: "Kobbie Mainoo", age: 19, position: "CM", nationality: "England", club: "Man United", overall: 78, potential: 89, pace: 70, shooting: 68, passing: 76, dribbling: 79, defending: 74, physical: 72 },
  { id: 11, name: "Warren Zaïre-Emery", age: 18, position: "CM", nationality: "France", club: "PSG", overall: 79, potential: 91, pace: 73, shooting: 70, passing: 78, dribbling: 80, defending: 73, physical: 70 },
  { id: 12, name: "Alejandro Garnacho", age: 20, position: "LW", nationality: "Argentina", club: "Man United", overall: 80, potential: 89, pace: 90, shooting: 74, passing: 72, dribbling: 83, defending: 32, physical: 65 },
  { id: 13, name: "Endrick", age: 18, position: "ST", nationality: "Brazil", club: "Real Madrid", overall: 76, potential: 91, pace: 85, shooting: 78, passing: 62, dribbling: 79, defending: 28, physical: 74 },
  { id: 14, name: "Arda Güler", age: 19, position: "CAM", nationality: "Turkey", club: "Real Madrid", overall: 79, potential: 90, pace: 71, shooting: 78, passing: 80, dribbling: 86, defending: 26, physical: 51 },
  { id: 15, name: "Mathys Tel", age: 19, position: "ST", nationality: "France", club: "Bayern", overall: 77, potential: 89, pace: 88, shooting: 75, passing: 67, dribbling: 79, defending: 30, physical: 68 },
];

// Legendary players for hero section
export const legendaryPlayers = [
  { id: 100, name: "Lionel Messi", age: 37, position: "RW", nationality: "Argentina", club: "Inter Miami", overall: 88, potential: 88, pace: 76, shooting: 89, passing: 90, dribbling: 94, defending: 34, physical: 65, image: "messi" },
  { id: 101, name: "Cristiano Ronaldo", age: 39, position: "ST", nationality: "Portugal", club: "Al Nassr", overall: 85, potential: 85, pace: 81, shooting: 92, passing: 78, dribbling: 84, defending: 34, physical: 77, image: "ronaldo" },
  { id: 102, name: "Erling Haaland", age: 24, position: "ST", nationality: "Norway", club: "Man City", overall: 91, potential: 94, pace: 89, shooting: 94, passing: 66, dribbling: 80, defending: 45, physical: 88, image: "haaland" },
];

// Superstar benchmark (average of top players)
export const superstarBenchmark = {
  pace: 85,
  shooting: 88,
  passing: 80,
  dribbling: 88,
  defending: 45,
  physical: 75,
};

export function calculateSuperstarProbability(player: Partial<Player>): PredictionResult {
  const stats = {
    pace: player.pace || 50,
    shooting: player.shooting || 50,
    passing: player.passing || 50,
    dribbling: player.dribbling || 50,
    defending: player.defending || 50,
    physical: player.physical || 50,
  };

  const overall = player.overall || 70;
  const potential = player.potential || 75;
  const age = player.age || 20;

  // Calculate base probability
  let probability = 0;
  
  // Potential is the biggest factor (40%)
  probability += (potential / 99) * 40;
  
  // Current overall (25%)
  probability += (overall / 99) * 25;
  
  // Stats comparison to benchmark (25%)
  const statKeys = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physical'] as const;
  let statScore = 0;
  statKeys.forEach(key => {
    const ratio = stats[key] / superstarBenchmark[key];
    statScore += Math.min(ratio, 1.2); // Cap at 120%
  });
  probability += (statScore / 7.2) * 25;
  
  // Youth bonus (10%)
  if (age <= 21) {
    probability += 10;
  } else if (age <= 23) {
    probability += 5;
  }

  // Clamp probability
  probability = Math.min(Math.max(probability, 5), 98);

  // Calculate confidence
  const confidence = Math.min(85 + (overall - 70) * 0.5, 98);

  // Determine tier
  let tier = "Prospect";
  if (probability >= 90) tier = "Generational Talent 🌟";
  else if (probability >= 80) tier = "Future Superstar 🔥";
  else if (probability >= 70) tier = "Elite Potential ⚡";
  else if (probability >= 60) tier = "Strong Prospect 💪";
  else if (probability >= 50) tier = "Developing Talent 📈";

  // Generate comparisons
  const comparisons = [
    {
      stat: "pace",
      label: "Pace",
      emoji: "💨",
      comparison: stats.pace >= 90 ? "Mbappé Tier" : stats.pace >= 80 ? "World Class" : stats.pace >= 70 ? "Above Average" : "Developing"
    },
    {
      stat: "shooting",
      label: "Shooting",
      emoji: "⚽",
      comparison: stats.shooting >= 88 ? "Haaland Level" : stats.shooting >= 80 ? "Elite Finisher" : stats.shooting >= 70 ? "Clinical" : "Improving"
    },
    {
      stat: "dribbling",
      label: "Dribbling",
      emoji: "🎯",
      comparison: stats.dribbling >= 90 ? "Messi DNA" : stats.dribbling >= 80 ? "Silky Skills" : stats.dribbling >= 70 ? "Technical" : "Work in Progress"
    },
    {
      stat: "passing",
      label: "Passing",
      emoji: "🎯",
      comparison: stats.passing >= 85 ? "Playmaker" : stats.passing >= 75 ? "Vision" : stats.passing >= 65 ? "Reliable" : "Basic"
    },
  ];

  return {
    probability: Math.round(probability),
    confidence: Math.round(confidence),
    tier,
    comparisons,
    benchmarkStats: [superstarBenchmark.pace, superstarBenchmark.shooting, superstarBenchmark.passing, superstarBenchmark.dribbling, superstarBenchmark.defending, superstarBenchmark.physical],
    userStats: [stats.pace, stats.shooting, stats.passing, stats.dribbling, stats.defending, stats.physical],
  };
}

export function searchPlayers(query: string): Player[] {
  if (!query.trim()) return [];
  const lowerQuery = query.toLowerCase();
  return mockPlayers.filter(p => 
    p.name.toLowerCase().includes(lowerQuery) ||
    p.club.toLowerCase().includes(lowerQuery) ||
    p.nationality.toLowerCase().includes(lowerQuery)
  ).slice(0, 8);
}
