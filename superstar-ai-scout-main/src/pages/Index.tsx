import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Trophy, Zap } from "lucide-react";
import { StadiumParticles } from "@/components/StadiumParticles";
import { PlayerCard } from "@/components/PlayerCard";
import { PlayerFilter } from "@/components/PlayerFilter";
import { getHighlights, type HighlightPlayer } from "@/lib/api";

export default function Index() {
  const navigate = useNavigate();
  const [legends, setLegends] = useState<HighlightPlayer[]>([]);
  const [prospects, setProspects] = useState<HighlightPlayer[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await getHighlights();
        setLegends(data.legends);
        setProspects(data.prospects);
      } catch (e) {
        console.error("Failed to load highlight players", e);
      }
    };

    load();
  }, []);

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
              Explore FIFA 21's
              <br />
              <span className="gradient-text">Future Superstars</span>
            </h2>
            <p className="mx-auto max-w-xl text-lg text-muted-foreground">
              Use our AI-powered scout to browse FIFA 21 players by country,
              position and age, and see who has the superstar DNA.
            </p>
          </motion.div>

          {/* Hero Player Cards */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
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
              {legends.map((player, index) => (
                <motion.div
                  key={player.player_id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 + index * 0.15 }}
                  className="w-[160px] md:w-[180px]"
                >
                  <PlayerCard
                    name={player.name}
                    position={player.position}
                    overall={player.overall}
                    potential={player.potential}
                    willBeSuperstar={player.will_be_superstar}
                    superstarLabel={player.superstar_label}
                    onClick={() => navigate(`/player/${player.player_id}`)}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Popular Prospects */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className="mt-16 w-full"
          >
            <div className="mb-4 flex items-center justify-center gap-2">
              <Zap className="h-4 w-4 text-primary" />
              <span className="font-display text-sm uppercase tracking-wider text-muted-foreground">
                Popular Young Prospects
              </span>
              <Zap className="h-4 w-4 text-primary" />
            </div>
            <div className="flex flex-wrap justify-center gap-4 md:gap-6">
              {prospects.map((player, index) => (
                <motion.div
                  key={player.player_id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 1.1 + index * 0.05 }}
                  className="w-[150px] md:w-[170px]"
                >
                  <PlayerCard
                    name={player.name}
                    position={player.position}
                    overall={player.overall}
                    potential={player.potential}
                    willBeSuperstar={player.will_be_superstar}
                    superstarLabel={player.superstar_label}
                    onClick={() => navigate(`/player/${player.player_id}`)}
                  />
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* FIFA 21 Filter Section */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
            className="w-full"
          >
            <PlayerFilter />
          </motion.div>
        </section>
      </main>
    </div>
  );
}
