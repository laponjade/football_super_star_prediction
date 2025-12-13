import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, Star, Share2, Trophy, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StadiumParticles } from "@/components/StadiumParticles";
import { RadarChart } from "@/components/RadarChart";
import { ProbabilityGauge } from "@/components/ProbabilityGauge";
import { StatBreakdown } from "@/components/StatBreakdown";
import { mockPlayers, legendaryPlayers, calculateSuperstarProbability } from "@/lib/players";
import { toast } from "@/hooks/use-toast";

export default function PlayerDetail() {
  const { name } = useParams<{ name: string }>();
  const navigate = useNavigate();

  const decodedName = decodeURIComponent(name || "");
  const allPlayers = [...mockPlayers, ...legendaryPlayers];
  const player = allPlayers.find(
    (p) => p.name.toLowerCase() === decodedName.toLowerCase()
  );

  if (!player) {
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

  const result = calculateSuperstarProbability(player);

  const handleShare = () => {
    const text = `Check out ${player.name}! ${result.probability}% Superstar Probability 🧬⚽ #SuperstarScoutAI`;
    navigator.clipboard.writeText(text);
    toast({
      title: "Copied to clipboard!",
      description: "Share this player's profile",
    });
  };

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
                {/* Top row - Rating & Stars */}
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
                    <div className="flex gap-0.5">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          size={16}
                          className={
                            i < Math.floor(result.probability / 20)
                              ? "fill-stadium-gold text-stadium-gold"
                              : "text-muted"
                          }
                        />
                      ))}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      POT {player.potential}
                    </span>
                  </div>
                </div>

                {/* Player Avatar */}
                <div className="mx-auto mb-4 flex h-32 w-32 items-center justify-center rounded-full border-4 border-primary/30 bg-gradient-to-br from-primary/20 to-stadium-gold/20">
                  <span className="font-display text-4xl font-bold text-foreground">
                    {player.name.split(" ").map((n) => n[0]).join("")}
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
                      <p className="text-xs text-muted-foreground">{stat.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Analysis Section */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Probability Gauge */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center gap-2">
              <Trophy className="h-5 w-5 text-stadium-gold" />
              <h2 className="font-display text-lg font-semibold text-foreground">
                Superstar Probability
              </h2>
            </div>
            <div className="flex justify-center">
              <ProbabilityGauge
                probability={result.probability}
                tier={result.tier}
              />
            </div>
          </motion.div>

          {/* Radar Chart */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur-sm"
          >
            <div className="mb-4 flex items-center gap-2">
              <Zap className="h-5 w-5 text-primary" />
              <h2 className="font-display text-lg font-semibold text-foreground">
                Stats vs Superstar Benchmark
              </h2>
            </div>
            <div className="flex justify-center">
              <RadarChart
                userStats={result.userStats}
                benchmarkStats={result.benchmarkStats}
              />
            </div>
          </motion.div>
        </div>

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
          <StatBreakdown comparisons={result.comparisons} />
        </motion.div>

        {/* Scout Quote */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-6 rounded-xl border border-primary/30 bg-primary/5 p-6 text-center"
        >
          <p className="font-body text-lg italic text-foreground">
            "{result.probability >= 80
              ? "This kid's got superstar written all over them! 🌟"
              : result.probability >= 60
              ? "Serious potential here - one to watch! 👀"
              : "A developing talent with room to grow 📈"}"
          </p>
          <p className="mt-2 text-sm text-muted-foreground">
            — Superstar Scout AI Analysis
          </p>
        </motion.div>

        {/* CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mt-8 text-center"
        >
          <Button
            size="lg"
            onClick={() => navigate("/predict")}
            className="gap-2 rounded-full bg-primary font-display font-semibold text-primary-foreground shadow-neon"
          >
            Create Your Own Player
          </Button>
        </motion.div>
      </main>
    </div>
  );
}
