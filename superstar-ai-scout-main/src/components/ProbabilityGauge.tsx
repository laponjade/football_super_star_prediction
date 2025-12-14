import { motion } from "framer-motion";

interface ProbabilityGaugeProps {
  probability: number;
  tier: string;
  size?: number;
}

export function ProbabilityGauge({ probability, tier, size = 200 }: ProbabilityGaugeProps) {
  const strokeWidth = 12;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = (probability / 100) * circumference;

  const getGradientColors = () => {
    if (probability >= 80) return ["#00ff88", "#00cc6a"];
    if (probability >= 60) return ["#ffd700", "#ffaa00"];
    return ["#ff6b6b", "#ff4757"];
  };

  const [color1, color2] = getGradientColors();

  return (
    <div className="relative flex flex-col items-center">
      <svg width={size} height={size} className="rotate-[-90deg]">
        <defs>
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color1} />
            <stop offset="100%" stopColor={color2} />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="hsl(var(--muted))"
          strokeWidth={strokeWidth}
          opacity={0.3}
        />

        {/* Progress arc */}
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="url(#gaugeGradient)"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: circumference - progress }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          filter="url(#glow)"
        />
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.8, type: "spring", stiffness: 200 }}
          className="font-display text-4xl font-bold text-primary text-glow"
        >
          {probability}%
        </motion.span>
        <span className="text-xs uppercase text-muted-foreground">Superstar</span>
      </div>

      {/* Tier badge */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2 }}
        className="mt-4 rounded-full border border-primary/50 bg-primary/10 px-4 py-2"
      >
        <span className="font-display text-sm font-semibold text-primary">{tier}</span>
      </motion.div>
    </div>
  );
}
