import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Sparkles, Trophy, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { StadiumParticles } from "@/components/StadiumParticles";
import { PlayerSearch } from "@/components/PlayerSearch";
import { PlayerCard } from "@/components/PlayerCard";
import { legendaryPlayers, calculateSuperstarProbability } from "@/lib/players";

export default function Index() {
  const navigate = useNavigate();

  const heroPlayers = legendaryPlayers.map((player) => ({
    ...player,
    probability: calculateSuperstarProbability(player).probability,
  }));

  return (
    <div className="relative min-h-screen overflow-hidden bg-background">
      <StadiumParticles />

      {/* Stadium glow effect */}
      <div className="pointer-events-none absolute inset-0 stadium-glow" />

      {/* Floodlight sweep */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="animate-floodlight absolute -left-1/2 top-0 h-full w-1/2 bg-gradient-to-r from-transparent via-primary/5 to-transparent" />
      </div>

      <main className="relative z-10">
        {/* Hero Section */}
        <section className="container flex min-h-screen flex-col items-center justify-center px-4 py-12">
          {/* Logo/Brand */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8 flex items-center gap-3"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/20 border border-primary/30">
              <Trophy className="h-6 w-6 text-primary" />
            </div>
            <h1 className="font-display text-2xl font-bold tracking-wider text-foreground">
              SUPERSTAR SCOUT <span className="text-primary">AI</span>
            </h1>
          </motion.div>

          {/* Main Headline */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-8 text-center"
          >
            <h2 className="mb-4 font-display text-4xl font-bold leading-tight text-foreground md:text-6xl lg:text-7xl">
              Find Football's
              <br />
              <span className="gradient-text">Next Legend</span>
            </h2>
            <p className="mx-auto max-w-xl text-lg text-muted-foreground">
              AI-powered scouting that predicts which young players will become
              the next Mbappé, Haaland, or Messi 🧬
            </p>
          </motion.div>

          {/* Search Bar */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 }}
            className="mb-12 w-full max-w-2xl"
          >
            <PlayerSearch />
          </motion.div>

          {/* CTA Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mb-16"
          >
            <Button
              size="lg"
              onClick={() => navigate("/predict")}
              className="group h-14 gap-2 rounded-full bg-primary px-8 font-display text-lg font-semibold text-primary-foreground shadow-neon transition-all hover:bg-primary/90 hover:shadow-neon-lg"
            >
              <Sparkles className="h-5 w-5" />
              Predict Future
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </Button>
          </motion.div>

          {/* Hero Player Cards */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 }}
            className="w-full"
          >
            <div className="mb-6 flex items-center justify-center gap-2">
              <Zap className="h-4 w-4 text-stadium-gold" />
              <span className="font-display text-sm uppercase tracking-wider text-muted-foreground">
                Legendary Superstars
              </span>
              <Zap className="h-4 w-4 text-stadium-gold" />
            </div>

            <div className="flex flex-wrap justify-center gap-4 md:gap-6">
              {heroPlayers.map((player, index) => (
                <motion.div
                  key={player.id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1 + index * 0.15 }}
                  className="w-[160px] md:w-[180px]"
                >
                  <PlayerCard
                    name={player.name}
                    position={player.position}
                    overall={player.overall}
                    probability={player.probability}
                    onClick={() => navigate("/predict")}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Stats bar */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.5 }}
            className="mt-16 flex flex-wrap justify-center gap-8 md:gap-16"
          >
            {[
              { label: "Players Analyzed", value: "10K+" },
              { label: "Accuracy Rate", value: "94%" },
              { label: "Future Stars Found", value: "847" },
            ].map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="font-display text-2xl font-bold text-primary text-glow md:text-3xl">
                  {stat.value}
                </p>
                <p className="text-sm text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </motion.div>
        </section>
      </main>
    </div>
  );
}
