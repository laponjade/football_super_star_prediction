import { motion } from "framer-motion";

interface StatBreakdownProps {
  comparisons: {
    stat: string;
    label: string;
    emoji: string;
    comparison: string;
  }[];
}

export function StatBreakdown({ comparisons }: StatBreakdownProps) {
  return (
    <div className="grid grid-cols-2 gap-3">
      {comparisons.map((comp, index) => (
        <motion.div
          key={comp.stat}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.5 + index * 0.1 }}
          className="rounded-lg border border-border bg-card/50 p-3 backdrop-blur-sm"
        >
          <div className="mb-1 flex items-center gap-2">
            <span className="text-lg">{comp.emoji}</span>
            <span className="text-xs font-medium uppercase text-muted-foreground">
              {comp.label}
            </span>
          </div>
          <p className="font-display text-sm font-semibold text-foreground">
            {comp.comparison}
          </p>
        </motion.div>
      ))}
    </div>
  );
}
