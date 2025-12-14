import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface FireworksProps {
  active: boolean;
  onComplete?: () => void;
}

interface Spark {
  id: number;
  x: number;
  y: number;
  color: string;
  delay: number;
}

export function Fireworks({ active, onComplete }: FireworksProps) {
  const [sparks, setSparks] = useState<Spark[]>([]);

  useEffect(() => {
    if (!active) {
      setSparks([]);
      return;
    }

    const colors = ["#00ff88", "#ffd700", "#ff6b6b", "#00d4ff"];
    const newSparks: Spark[] = [];

    for (let i = 0; i < 20; i++) {
      newSparks.push({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 60 + 20,
        color: colors[Math.floor(Math.random() * colors.length)],
        delay: Math.random() * 0.5,
      });
    }

    setSparks(newSparks);

    const timer = setTimeout(() => {
      onComplete?.();
    }, 2000);

    return () => clearTimeout(timer);
  }, [active, onComplete]);

  return (
    <AnimatePresence>
      {active && (
        <div className="pointer-events-none fixed inset-0 z-50 overflow-hidden">
          {sparks.map((spark) => (
            <motion.div
              key={spark.id}
              initial={{ 
                left: `${spark.x}%`, 
                top: `${spark.y}%`,
                scale: 0,
                opacity: 1 
              }}
              animate={{ 
                scale: [0, 1.5, 0],
                opacity: [1, 1, 0]
              }}
              exit={{ opacity: 0 }}
              transition={{ 
                duration: 1,
                delay: spark.delay,
                ease: "easeOut"
              }}
              className="absolute"
              style={{
                width: 20,
                height: 20,
                borderRadius: "50%",
                background: `radial-gradient(circle, ${spark.color} 0%, transparent 70%)`,
                boxShadow: `0 0 20px ${spark.color}, 0 0 40px ${spark.color}`,
              }}
            />
          ))}
        </div>
      )}
    </AnimatePresence>
  );
}
