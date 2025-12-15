import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StadiumParticles } from "@/components/StadiumParticles";
import { StatBreakdown } from "@/components/StatBreakdown";
import {
  getPlayerById,
  predictPlayer,
  type PlayerSearchResult,
  type PredictionResponse,
} from "@/lib/api";
import { toast } from "@/hooks/use-toast";

export default function PlayerDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [player, setPlayer] = useState<PlayerSearchResult | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      if (!id) {
        setIsLoading(false);
        return;
      }

      const playerId = Number(id);
      if (Number.isNaN(playerId)) {
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);

        const p = await getPlayerById(playerId);
        if (!p) {
          toast({
            title: "Player not found",
            description: "Could not load player details.",
            variant: "destructive",
          });
          setIsLoading(false);
          return;
        }

        setPlayer(p);

        const pred = await predictPlayer({ player_id: playerId });
        setPrediction(pred);
      } catch (error: any) {
        console.error("Error loading player detail", error);
        toast({
          title: "Error",
          description:
            error?.message ||
            "Failed to load player details. Please try again.",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    load();
  }, [id, toast]);

  if (!player || !prediction) {
    if (isLoading) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-background">
          <p className="text-sm text-muted-foreground">Loading player...</p>
        </div>
      );
    }

    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <div className="text-center">
          <h1 className="mb-4 font-display text-2xl font-bold text-foreground">
            Player Not Found
          </h1>
          <Button onClick={() => navigate("/")} variant="outline">
            Go Home
          </Button>
        </div>
      </div>
    );
  }

  const result = prediction;

  const handleShare = () => {
    const text = `Check out ${player.name}! ${result.probability}% Superstar Probability 🧬⚽ #SuperstarScoutAI`;
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard!",
      description: "Share this player's profile",
    });
  };

  const comparisons = [
    {
      stat: "pace",
      label: "Pace",
      emoji: "💨",
      comparison:
        (player.pace || 50) >= 90
          ? "Mbappé Tier"
          : (player.pace || 50) >= 80
          ? "World Class"
          : (player.pace || 50) >= 70
          ? "Above Average"
          : "Developing",
    },
    {
      stat: "shooting",
      label: "Shooting",
      emoji: "⚽",
      comparison:
        (player.shooting || 50) >= 88
          ? "Haaland Level"
          : (player.shooting || 50) >= 80
          ? "Elite Finisher"
          : (player.shooting || 50) >= 70
          ? "Clinical"
          : "Improving",
    },
    {
      stat: "dribbling",
      label: "Dribbling",
      emoji: "🎯",
      comparison:
        (player.dribbling || 50) >= 90
          ? "Messi DNA"
          : (player.dribbling || 50) >= 80
          ? "Silky Skills"
          : (player.dribbling || 50) >= 70
          ? "Technical"
          : "Work in Progress",
    },
    {
      stat: "passing",
      label: "Passing",
      emoji: "🎯",
      comparison:
        (player.passing || 50) >= 85
          ? "Playmaker"
          : (player.passing || 50) >= 75
          ? "Vision"
          : (player.passing || 50) >= 65
          ? "Reliable"
          : "Basic",
    },
  ];

  return (
    <div className="relative min-h-screen bg-background">
      <StadiumParticles />
      <div className="pointer-events-none absolute inset-0 stadium-glow" />

      <main className="relative z-10 container px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-center justify-between"
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={() => navigate("/")}
            className="rounded-full border border-border hover:bg-primary/10 hover:border-primary/50"
          >
            <ArrowLeft className="h-5 w-5" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            onClick={handleShare}
            className="rounded-full border border-border hover:bg-primary/10 hover:border-primary/50"
          >
            <Share2 className="h-5 w-5" />
          </Button>
        </motion.div>

        {/* Player Card Hero */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <div className="mx-auto max-w-md overflow-hidden rounded-2xl border border-primary/30 bg-gradient-to-br from-card via-card to-secondary p-1 shadow-neon">
            <div className="relative rounded-xl bg-card/90 p-6 backdrop-blur-sm">
              {/* Glow effect */}
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/10 via-transparent to-stadium-gold/10" />

              <div className="relative">
                {/* Top row - Rating & Info */}
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex flex-col">
                    <span className="font-display text-5xl font-bold text-primary text-glow">
                      {player.overall}
                    </span>
                    <span className="text-sm uppercase text-muted-foreground">
                      {player.position}
                    </span>
                  </div>

                  <div className="flex flex-col items-end gap-1">
                    <span className="rounded-full bg-primary/15 px-3 py-1 text-xs font-medium text-primary">
                      POT {player.potential}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {result.superstar_label}
                    </span>
                  </div>
                </div>

                {/* Player Avatar */}
                <div className="mx-auto mb-4 flex h-32 w-32 items-center justify-center rounded-full border-4 border-primary/30 bg-gradient-to-br from-primary/20 to-stadium-gold/20">
                  <span className="font-display text-4xl font-bold text-foreground">
                    {player.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")}
                  </span>
                </div>

                {/* Name & Info */}
                <div className="mb-4 text-center">
                  <h1 className="font-display text-2xl font-bold uppercase tracking-wide text-foreground">
                    {player.name}
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    {player.club} • {player.nationality} • {player.age} years
                  </p>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-6 gap-2 text-center">
                  {[
                    { label: "PAC", value: player.pace },
                    { label: "SHO", value: player.shooting },
                    { label: "PAS", value: player.passing },
                    { label: "DRI", value: player.dribbling },
                    { label: "DEF", value: player.defending },
                    { label: "PHY", value: player.physical },
                  ].map((stat) => (
                    <div key={stat.label}>
                      <p className="font-display text-lg font-bold text-foreground">
                        {stat.value}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {stat.label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Breakdown */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="mt-6 rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm"
        >
          <h2 className="mb-4 font-display text-lg font-semibold text-foreground">
            Skill Analysis
          </h2>
          <StatBreakdown comparisons={comparisons} />
        </motion.div>
      </main>
    </div>
  );
}
