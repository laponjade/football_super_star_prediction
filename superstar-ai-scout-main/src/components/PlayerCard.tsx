import { motion } from "framer-motion";
import { Star } from "lucide-react";

interface PlayerCardProps {
  name: string;
  position: string;
  overall: number;
  probability: number;
  image?: string;
  onClick?: () => void;
}

export function PlayerCard({ name, position, overall, probability, onClick }: PlayerCardProps) {
  const firstName = name.split(" ")[0];
  const lastName = name.split(" ").slice(1).join(" ");

  return (
    <motion.div
      className="card-3d cursor-pointer"
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="card-3d-inner relative">
        {/* Card Background */}
        <div className="relative overflow-hidden rounded-xl border border-primary/30 bg-gradient-to-br from-card via-card to-secondary p-1 shadow-neon">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-transparent to-stadium-gold/10" />
          
          {/* Inner Card */}
          <div className="relative rounded-lg bg-card/90 p-4 backdrop-blur-sm">
            {/* Top Section - Rating */}
            <div className="mb-3 flex items-start justify-between">
              <div className="flex flex-col items-center">
                <span className="font-display text-3xl font-bold text-primary text-glow">
                  {overall}
                </span>
                <span className="text-xs uppercase text-muted-foreground">{position}</span>
              </div>
              <div className="flex gap-0.5">
                {[...Array(5)].map((_, i) => (
                  <Star
                    key={i}
                    size={12}
                    className={i < Math.floor(probability / 20) ? "fill-stadium-gold text-stadium-gold" : "text-muted"}
                  />
                ))}
              </div>
            </div>

            {/* Player Avatar Placeholder */}
            <div className="mx-auto mb-3 flex h-24 w-24 items-center justify-center rounded-full border-2 border-primary/30 bg-gradient-to-br from-primary/20 to-stadium-gold/20">
              <span className="font-display text-2xl font-bold text-foreground">
                {firstName[0]}{lastName?.[0] || ""}
              </span>
            </div>

            {/* Player Name */}
            <div className="mb-2 text-center">
              <p className="text-xs text-muted-foreground">{firstName}</p>
              <p className="font-display text-sm font-bold uppercase tracking-wide text-foreground">
                {lastName || firstName}
              </p>
            </div>

            {/* Probability Badge */}
            <motion.div
              className="mx-auto w-fit rounded-full border border-primary/50 bg-primary/20 px-3 py-1"
              animate={{ boxShadow: ["0 0 10px hsl(151 100% 50% / 0.3)", "0 0 20px hsl(151 100% 50% / 0.5)", "0 0 10px hsl(151 100% 50% / 0.3)"] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="font-display text-xs font-semibold text-primary">
                {probability}% Superstar 🔥
              </span>
            </motion.div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
