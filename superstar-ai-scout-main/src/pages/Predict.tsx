import { useState, useCallback, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Sparkles, Target, Share2, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StadiumParticles } from "@/components/StadiumParticles";
import { RadarChart } from "@/components/RadarChart";
import { ProbabilityGauge } from "@/components/ProbabilityGauge";
import { StatBreakdown } from "@/components/StatBreakdown";
import { Fireworks } from "@/components/Fireworks";
import { LoadingOverlay } from "@/components/LoadingOverlay";
import { type PredictionResult, superstarBenchmark } from "@/lib/players";
import { searchPlayers, predictPlayer, getPlayerById, type PlayerSearchResult } from "@/lib/api";
import { toast } from "@/hooks/use-toast";

const positions = [
  { value: "ST", label: "Striker (ST)" },
  { value: "CF", label: "Center Forward (CF)" },
  { value: "LW", label: "Left Wing (LW)" },
  { value: "RW", label: "Right Wing (RW)" },
  { value: "CAM", label: "Attacking Mid (CAM)" },
  { value: "CM", label: "Central Mid (CM)" },
  { value: "CDM", label: "Defensive Mid (CDM)" },
  { value: "LB", label: "Left Back (LB)" },
  { value: "RB", label: "Right Back (RB)" },
  { value: "CB", label: "Center Back (CB)" },
  { value: "GK", label: "Goalkeeper (GK)" },
];

export default function Predict() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [showFireworks, setShowFireworks] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerSearchResult | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<PlayerSearchResult[]>([]);
  const [showSearchResults, setShowSearchResults] = useState(false);
  const [isSearching, setIsSearching] = useState(false);

  const [formData, setFormData] = useState<{
    age: number;
    position: string;
    overall: number;
    potential: number;
    pace: number;
    shooting: number;
    passing: number;
    dribbling: number;
    defending: number;
    physical: number;
  }>({
    age: 19,
    position: "ST",
    overall: 75,
    potential: 85,
    pace: 80,
    shooting: 75,
    passing: 70,
    dribbling: 78,
    defending: 40,
    physical: 70,
  });

  // Helper function to predict from player_id
  const handlePredictFromPlayerId = useCallback(async (playerId: number) => {
    setIsLoading(true);
    setResult(null);

    try {
      const predictionResponse = await predictPlayer({
        player_id: playerId,
      });

      // We need to get player details for the stats display
      // For now, use form defaults - we'll improve this later
      const prediction: PredictionResult = {
        probability: predictionResponse.probability,
        confidence: predictionResponse.confidence,
        tier: predictionResponse.tier,
        comparisons: [
          {
            stat: "pace",
            label: "Pace",
            emoji: "💨",
            comparison: "Analyzed"
          },
          {
            stat: "shooting",
            label: "Shooting",
            emoji: "⚽",
            comparison: "Analyzed"
          },
          {
            stat: "dribbling",
            label: "Dribbling",
            emoji: "🎯",
            comparison: "Analyzed"
          },
          {
            stat: "passing",
            label: "Passing",
            emoji: "🎯",
            comparison: "Analyzed"
          },
        ],
        benchmarkStats: [
          superstarBenchmark.pace,
          superstarBenchmark.shooting,
          superstarBenchmark.passing,
          superstarBenchmark.dribbling,
          superstarBenchmark.defending,
          superstarBenchmark.physical
        ],
        userStats: [
          80, 80, 80, 80, 50, 70 // Default stats - we'll improve this
        ],
      };

      setResult(prediction);

      if (prediction.probability >= 70) {
        setShowFireworks(true);
      }

      toast({
        title: "🧬 DNA Analysis Complete!",
        description: `This player has ${prediction.probability >= 80 ? "superstar" : prediction.probability >= 60 ? "serious" : "some"} potential!`,
      });
    } catch (error: any) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction Error",
        description: error.message || "Failed to predict. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [toast]);

  // Handle player_id from URL params - auto-load, fill form, and predict
  useEffect(() => {
    const playerIdParam = searchParams.get('player_id');
    if (playerIdParam) {
      const playerId = parseInt(playerIdParam, 10);
      if (!isNaN(playerId)) {
        // Fetch player details and fill form
        const loadPlayerAndPredict = async () => {
          setIsLoading(true);
          try {
            // Get full player data
            const player = await getPlayerById(playerId);
            if (player) {
              // Set selected player
              setSelectedPlayer(player);
              setSearchQuery(player.name);
              
              // Fill form with player's actual data
              setFormData({
                age: player.age || 19,
                position: player.position || "ST",
                overall: player.overall || 75,
                potential: player.potential || 85,
                pace: player.pace || 50,
                shooting: player.shooting || 50,
                passing: player.passing || 50,
                dribbling: player.dribbling || 50,
                defending: player.defending || 50,
                physical: player.physical || 50,
              });
              
              // Automatically trigger prediction
              const predictionResponse = await predictPlayer({
                player_id: playerId,
              });
              
              // Convert API response to PredictionResult format
              const prediction: PredictionResult = {
                probability: predictionResponse.probability,
                confidence: predictionResponse.confidence,
                tier: predictionResponse.tier,
                comparisons: [
                  {
                    stat: "pace",
                    label: "Pace",
                    emoji: "💨",
                    comparison: (player.pace || 50) >= 90 ? "Mbappé Tier" : (player.pace || 50) >= 80 ? "World Class" : (player.pace || 50) >= 70 ? "Above Average" : "Developing"
                  },
                  {
                    stat: "shooting",
                    label: "Shooting",
                    emoji: "⚽",
                    comparison: (player.shooting || 50) >= 88 ? "Haaland Level" : (player.shooting || 50) >= 80 ? "Elite Finisher" : (player.shooting || 50) >= 70 ? "Clinical" : "Improving"
                  },
                  {
                    stat: "dribbling",
                    label: "Dribbling",
                    emoji: "🎯",
                    comparison: (player.dribbling || 50) >= 90 ? "Messi DNA" : (player.dribbling || 50) >= 80 ? "Silky Skills" : (player.dribbling || 50) >= 70 ? "Technical" : "Work in Progress"
                  },
                  {
                    stat: "passing",
                    label: "Passing",
                    emoji: "🎯",
                    comparison: (player.passing || 50) >= 85 ? "Playmaker" : (player.passing || 50) >= 75 ? "Vision" : (player.passing || 50) >= 65 ? "Reliable" : "Basic"
                  },
                ],
                benchmarkStats: [
                  superstarBenchmark.pace,
                  superstarBenchmark.shooting,
                  superstarBenchmark.passing,
                  superstarBenchmark.dribbling,
                  superstarBenchmark.defending,
                  superstarBenchmark.physical
                ],
                userStats: [
                  player.pace || 50,
                  player.shooting || 50,
                  player.passing || 50,
                  player.dribbling || 50,
                  player.defending || 50,
                  player.physical || 50,
                ],
              };
              
              setResult(prediction);
              
              if (prediction.probability >= 70) {
                setShowFireworks(true);
              }
              
              toast({
                title: "🧬 DNA Analysis Complete!",
                description: `${player.name} has ${prediction.probability >= 80 ? "superstar" : prediction.probability >= 60 ? "serious" : "some"} potential!`,
              });
            } else {
              toast({
                title: "Player Not Found",
                description: "Could not load player details. Please try again.",
                variant: "destructive",
              });
            }
          } catch (error: any) {
            console.error("Error loading player:", error);
            toast({
              title: "Error",
              description: error.message || "Failed to load player. Please try again.",
              variant: "destructive",
            });
          } finally {
            setIsLoading(false);
            // Clear the URL param
            setSearchParams({});
          }
        };
        
        loadPlayerAndPredict();
      }
    }
  }, [searchParams, setSearchParams, toast]);

  // Handle player search with debounce
  useEffect(() => {
    const trimmedQuery = searchQuery.trim();
    
    // Require at least 3 characters before searching
    if (!trimmedQuery || trimmedQuery.length < 3) {
      setSearchResults([]);
      setShowSearchResults(false);
      return;
    }

    const timeoutId = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await searchPlayers(trimmedQuery, 10);
        setSearchResults(results);
        setShowSearchResults(true);
      } catch (error) {
        console.error("Search error:", error);
        toast({
          title: "Search Error",
          description: "Failed to search players. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsSearching(false);
      }
    }, 300);

    return () => clearTimeout(timeoutId);
  }, [searchQuery, toast]);

  const handleSelectPlayer = (player: PlayerSearchResult) => {
    setSelectedPlayer(player);
    setSearchQuery(player.name);
    setShowSearchResults(false);
    
    // Fill form with player's actual data
    setFormData({
      age: player.age || 19,
      position: player.position || "ST",
      overall: player.overall || 75,
      potential: player.potential || 85,
      pace: player.pace || 50,
      shooting: player.shooting || 50,
      passing: player.passing || 50,
      dribbling: player.dribbling || 50,
      defending: player.defending || 50,
      physical: player.physical || 50,
    });
    
    toast({
      title: "Player Selected",
      description: `${player.name}'s information has been loaded into the form.`,
    });
  };

  const handleClearPlayer = () => {
    setSelectedPlayer(null);
    setSearchQuery("");
    setSearchResults([]);
  };

  const updateField = (field: string, value: number | string) => {
    // Ensure numeric fields are properly converted to numbers
    const processedValue = typeof value === 'number' ? value : 
                          (field === 'position' ? value : Number(value) || 0);
    
    setFormData((prev) => ({ 
      ...prev, 
      [field]: processedValue 
    }));
    
    // Clear selected player when form is manually edited
    if (selectedPlayer) {
      setSelectedPlayer(null);
    }
  };

  const handlePredict = useCallback(async () => {
    console.log('handlePredict called', { selectedPlayer, formData });
    setIsLoading(true);
    setResult(null);

    try {
      let predictionResponse;

      if (selectedPlayer) {
        // Mode 1: Predict from existing player
        console.log('Predicting from selected player:', selectedPlayer.player_id);
        predictionResponse = await predictPlayer({
          player_id: selectedPlayer.player_id,
        });
      } else {
        // Mode 2: Predict from form input
        // Validate and ensure all values are properly formatted
        const validatedFormData = {
          age: Number(formData.age) || 20,
          position: String(formData.position || 'ST').trim(),
          overall: Number(formData.overall) || 75,
          potential: Number(formData.potential) || 85,
          pace: Number(formData.pace) || 50,
          shooting: Number(formData.shooting) || 50,
          passing: Number(formData.passing) || 50,
          dribbling: Number(formData.dribbling) || 50,
          defending: Number(formData.defending) || 50,
          physical: Number(formData.physical) || 50,
        };

        // Validate ranges
        if (validatedFormData.age < 16 || validatedFormData.age > 50) {
          throw new Error('Age must be between 16 and 50');
        }
        if (validatedFormData.overall < 40 || validatedFormData.overall > 99) {
          throw new Error('Overall must be between 40 and 99');
        }
        if (validatedFormData.potential < 40 || validatedFormData.potential > 99) {
          throw new Error('Potential must be between 40 and 99');
        }
        if (validatedFormData.pace < 1 || validatedFormData.pace > 99) {
          throw new Error('Pace must be between 1 and 99');
        }
        if (validatedFormData.shooting < 1 || validatedFormData.shooting > 99) {
          throw new Error('Shooting must be between 1 and 99');
        }
        if (validatedFormData.passing < 1 || validatedFormData.passing > 99) {
          throw new Error('Passing must be between 1 and 99');
        }
        if (validatedFormData.dribbling < 1 || validatedFormData.dribbling > 99) {
          throw new Error('Dribbling must be between 1 and 99');
        }
        if (validatedFormData.defending < 1 || validatedFormData.defending > 99) {
          throw new Error('Defending must be between 1 and 99');
        }
        if (validatedFormData.physical < 1 || validatedFormData.physical > 99) {
          throw new Error('Physical must be between 1 and 99');
        }
        if (!validatedFormData.position) {
          throw new Error('Position is required');
        }

        console.log('Predicting from validated form data:', validatedFormData);
        predictionResponse = await predictPlayer(validatedFormData);
      }
      
      console.log('Prediction response:', predictionResponse);

      // Convert API response to PredictionResult format
      const prediction: PredictionResult = {
        probability: predictionResponse.probability,
        confidence: predictionResponse.confidence,
        tier: predictionResponse.tier,
        comparisons: [
          {
            stat: "pace",
            label: "Pace",
            emoji: "💨",
            comparison: formData.pace >= 90 ? "Mbappé Tier" : formData.pace >= 80 ? "World Class" : formData.pace >= 70 ? "Above Average" : "Developing"
          },
          {
            stat: "shooting",
            label: "Shooting",
            emoji: "⚽",
            comparison: formData.shooting >= 88 ? "Haaland Level" : formData.shooting >= 80 ? "Elite Finisher" : formData.shooting >= 70 ? "Clinical" : "Improving"
          },
          {
            stat: "dribbling",
            label: "Dribbling",
            emoji: "🎯",
            comparison: formData.dribbling >= 90 ? "Messi DNA" : formData.dribbling >= 80 ? "Silky Skills" : formData.dribbling >= 70 ? "Technical" : "Work in Progress"
          },
          {
            stat: "passing",
            label: "Passing",
            emoji: "🎯",
            comparison: formData.passing >= 85 ? "Playmaker" : formData.passing >= 75 ? "Vision" : formData.passing >= 65 ? "Reliable" : "Basic"
          },
        ],
        benchmarkStats: [
          superstarBenchmark.pace,
          superstarBenchmark.shooting,
          superstarBenchmark.passing,
          superstarBenchmark.dribbling,
          superstarBenchmark.defending,
          superstarBenchmark.physical
        ],
        userStats: [
          formData.pace,
          formData.shooting,
          formData.passing,
          formData.dribbling,
          formData.defending,
          formData.physical
        ],
      };

      setResult(prediction);

      if (prediction.probability >= 70) {
        setShowFireworks(true);
      }

      toast({
        title: "🧬 DNA Analysis Complete!",
        description: `This kid's got ${prediction.probability >= 80 ? "superstar" : prediction.probability >= 60 ? "serious" : "some"} potential written all over them!`,
      });
    } catch (error: any) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction Error",
        description: error.message || "Failed to predict. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  }, [formData, selectedPlayer, toast]);

  const handleShare = () => {
    if (!result) return;
    
    const text = `I found a potential superstar! ${result.probability}% Superstar DNA 🧬⚽`;
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard!",
      description: "Share this discovery with your friends",
    });
  };

  return (
    <div className="relative min-h-screen bg-background">
      <StadiumParticles />
      <LoadingOverlay isLoading={isLoading} />
      <Fireworks active={showFireworks} onComplete={() => setShowFireworks(false)} />

      <div className="pointer-events-none absolute inset-0 stadium-glow" />

      <main className="relative z-10 container px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-center gap-4"
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/")}
            className="rounded-full border border-border hover:bg-primary/10 hover:border-primary/50"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <h1 className="font-display text-2xl font-bold text-foreground md:text-3xl">
              Player <span className="text-primary">Predictor</span>
            </h1>
            <p className="text-sm text-muted-foreground">
              Create a custom player profile and discover their superstar potential
            </p>
          </div>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Form Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm">
              <div className="mb-6 flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                <h2 className="font-display text-lg font-semibold text-foreground">
                  Player Profile
                </h2>
              </div>

              <div className="space-y-6">
                {/* Player Search */}
                <div>
                  <Label className="text-muted-foreground">Search Player (Optional)</Label>
                  <div className="mt-2 relative">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        type="text"
                        placeholder="Search by name, club, or nationality..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        onFocus={() => {
                          if (searchQuery && searchResults.length > 0) {
                            setShowSearchResults(true);
                          }
                        }}
                        onBlur={() => {
                          // Delay to allow click events to fire
                          setTimeout(() => setShowSearchResults(false), 200);
                        }}
                        className="pl-10 pr-10"
                      />
                      {searchQuery && (
                        <button
                          onClick={handleClearPlayer}
                          className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      )}
                    </div>
                    {showSearchResults && searchResults.length > 0 && (
                      <div 
                        className="absolute z-50 w-full mt-1 bg-card border border-border rounded-lg shadow-lg max-h-60 overflow-y-auto"
                        onMouseDown={(e) => e.preventDefault()}
                      >
                        {searchResults.map((player) => (
                          <button
                            key={player.player_id}
                            onClick={() => handleSelectPlayer(player)}
                            className="w-full text-left px-4 py-3 hover:bg-accent transition-colors border-b border-border last:border-b-0"
                          >
                            <div className="font-semibold">{player.name}</div>
                            <div className="text-sm text-muted-foreground">
                              {player.club} • {player.nationality} • {player.position}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                              OVR: {player.overall} | POT: {player.potential} | Age: {player.age}
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                    {searchQuery && searchQuery.length < 3 && (
                      <div className="absolute z-50 w-full mt-1 bg-card border border-border rounded-lg shadow-lg p-4 text-center text-sm text-muted-foreground">
                        Type at least 3 characters to search
                      </div>
                    )}
                    {isSearching && searchQuery.length >= 3 && (
                      <div className="absolute z-50 w-full mt-1 bg-card border border-border rounded-lg shadow-lg p-4 text-center text-sm text-muted-foreground">
                        Searching...
                      </div>
                    )}
                  </div>
                  {selectedPlayer && (
                    <div className="mt-2 p-3 bg-primary/10 rounded-lg border border-primary/20">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-semibold text-sm">{selectedPlayer.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {selectedPlayer.club} • {selectedPlayer.nationality}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleClearPlayer}
                          className="h-6 w-6 p-0"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Age & Position Row */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <Label className="text-muted-foreground">Age</Label>
                    <div className="mt-2 flex items-center gap-4">
                      <Slider
                        value={[formData.age]}
                        onValueChange={([v]) => updateField("age", v)}
                        min={16}
                        max={25}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-8 font-display text-lg font-bold text-primary">
                        {formData.age}
                      </span>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Position</Label>
                    <Select
                      value={formData.position}
                      onValueChange={(v) => updateField("position", v)}
                    >
                      <SelectTrigger className="mt-2 bg-input border-border">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-card border-border">
                        {positions.map((pos) => (
                          <SelectItem key={pos.value} value={pos.value}>
                            {pos.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Overall & Potential */}
                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <Label className="text-muted-foreground">Overall (1-99)</Label>
                    <div className="mt-2 flex items-center gap-4">
                      <Slider
                        value={[formData.overall]}
                        onValueChange={([v]) => updateField("overall", v)}
                        min={40}
                        max={99}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-8 font-display text-lg font-bold text-foreground">
                        {formData.overall}
                      </span>
                    </div>
                  </div>

                  <div>
                    <Label className="text-muted-foreground">Potential (1-99)</Label>
                    <div className="mt-2 flex items-center gap-4">
                      <Slider
                        value={[formData.potential]}
                        onValueChange={([v]) => updateField("potential", v)}
                        min={40}
                        max={99}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-8 font-display text-lg font-bold text-stadium-gold">
                        {formData.potential}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Stats */}
                <div className="space-y-4">
                  <Label className="text-muted-foreground">Key Stats</Label>
                  
                  {[
                    { key: "pace", label: "Pace", emoji: "💨" },
                    { key: "shooting", label: "Shooting", emoji: "⚽" },
                    { key: "passing", label: "Passing", emoji: "🎯" },
                    { key: "dribbling", label: "Dribbling", emoji: "✨" },
                    { key: "defending", label: "Defending", emoji: "🛡️" },
                    { key: "physical", label: "Physical", emoji: "💪" },
                  ].map(({ key, label, emoji }) => (
                    <div key={key} className="flex items-center gap-4">
                      <span className="w-24 text-sm text-muted-foreground">
                        {emoji} {label}
                      </span>
                      <Slider
                        value={[formData[key as keyof typeof formData] as number]}
                        onValueChange={([v]) => updateField(key, v)}
                        min={1}
                        max={99}
                        step={1}
                        className="flex-1"
                      />
                      <span className="w-8 text-right font-display text-sm font-bold text-foreground">
                        {formData[key as keyof typeof formData]}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Submit Button */}
                <Button
                  size="lg"
                  onClick={handlePredict}
                  disabled={isLoading}
                  className="w-full gap-2 rounded-full bg-primary font-display text-lg font-semibold text-primary-foreground shadow-neon transition-all hover:bg-primary/90 hover:shadow-neon-lg"
                >
                  <Sparkles className="h-5 w-5" />
                  Predict Superstar DNA
                </Button>
              </div>
            </div>
          </motion.div>

          {/* Results Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            {result ? (
              <div className="space-y-6">
                {/* Probability Gauge */}
                <div className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm">
                  <div className="flex flex-col items-center">
                    <ProbabilityGauge
                      probability={result.probability}
                      tier={result.tier}
                    />
                    
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 1.4 }}
                      className="mt-4 flex items-center gap-2"
                    >
                      <span className="rounded-full bg-primary/20 px-3 py-1 text-xs font-medium text-primary">
                        Beat {Math.round(result.probability * 0.95)}% of scouts
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={handleShare}
                        className="rounded-full"
                      >
                        <Share2 className="h-4 w-4" />
                      </Button>
                    </motion.div>
                  </div>
                </div>

                {/* Radar Chart */}
                <div className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm">
                  <h3 className="mb-4 text-center font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    Stats Comparison
                  </h3>
                  <div className="flex justify-center">
                    <RadarChart
                      userStats={result.userStats}
                      benchmarkStats={result.benchmarkStats}
                    />
                  </div>
                </div>

                {/* Breakdown Cards */}
                <div className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm">
                  <h3 className="mb-4 font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    Skill Breakdown
                  </h3>
                  <StatBreakdown comparisons={result.comparisons} />
                </div>
              </div>
            ) : (
              <div className="flex h-full min-h-[400px] items-center justify-center rounded-xl border border-dashed border-border bg-card/30 p-6">
                <div className="text-center">
                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                    <Target className="h-8 w-8 text-primary/50" />
                  </div>
                  <p className="font-display text-lg text-muted-foreground">
                    Create a player profile and click
                    <br />
                    <span className="text-primary">"Predict Superstar DNA"</span>
                  </p>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </main>

      {/* Mobile Sticky Button */}
      {!result && (
        <div className="fixed bottom-4 left-4 right-4 z-20 lg:hidden">
          <Button
            size="lg"
            onClick={handlePredict}
            disabled={isLoading}
            className="w-full gap-2 rounded-full bg-primary font-display text-lg font-semibold text-primary-foreground shadow-neon"
          >
            <Sparkles className="h-5 w-5" />
            Predict Superstar DNA
          </Button>
        </div>
      )}
    </div>
  );
}
