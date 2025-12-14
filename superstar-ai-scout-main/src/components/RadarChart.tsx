import { motion } from "framer-motion";
import { useMemo } from "react";

interface RadarChartProps {
  userStats: number[];
  benchmarkStats: number[];
  labels?: string[];
  size?: number;
}

export function RadarChart({ 
  userStats, 
  benchmarkStats, 
  labels = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"],
  size = 280 
}: RadarChartProps) {
  const center = size / 2;
  const radius = size * 0.38;

  const getPoint = (value: number, index: number, maxValue: number = 99) => {
    const angle = (Math.PI * 2 * index) / labels.length - Math.PI / 2;
    const normalizedRadius = (value / maxValue) * radius;
    return {
      x: center + Math.cos(angle) * normalizedRadius,
      y: center + Math.sin(angle) * normalizedRadius,
    };
  };

  const createPolygonPoints = (values: number[]) => {
    return values.map((v, i) => {
      const point = getPoint(v, i);
      return `${point.x},${point.y}`;
    }).join(" ");
  };

  const gridLevels = [0.25, 0.5, 0.75, 1];

  const labelPositions = useMemo(() => {
    return labels.map((_, i) => {
      const angle = (Math.PI * 2 * i) / labels.length - Math.PI / 2;
      return {
        x: center + Math.cos(angle) * (radius + 24),
        y: center + Math.sin(angle) * (radius + 24),
      };
    });
  }, [labels.length, center, radius]);

  return (
    <div className="relative">
      <svg width={size} height={size} className="overflow-visible">
        {/* Grid */}
        {gridLevels.map((level, i) => (
          <polygon
            key={i}
            points={labels.map((_, idx) => {
              const point = getPoint(99 * level, idx);
              return `${point.x},${point.y}`;
            }).join(" ")}
            fill="none"
            stroke="hsl(var(--border))"
            strokeWidth="1"
            opacity={0.5}
          />
        ))}

        {/* Axis lines */}
        {labels.map((_, i) => {
          const point = getPoint(99, i);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={point.x}
              y2={point.y}
              stroke="hsl(var(--border))"
              strokeWidth="1"
              opacity={0.3}
            />
          );
        })}

        {/* Benchmark polygon */}
        <motion.polygon
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          points={createPolygonPoints(benchmarkStats)}
          fill="hsl(var(--stadium-gold) / 0.15)"
          stroke="hsl(var(--stadium-gold))"
          strokeWidth="2"
          strokeDasharray="4 4"
        />

        {/* User stats polygon */}
        <motion.polygon
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, type: "spring", stiffness: 100 }}
          points={createPolygonPoints(userStats)}
          fill="hsl(var(--primary) / 0.3)"
          stroke="hsl(var(--primary))"
          strokeWidth="2"
          className="drop-shadow-[0_0_8px_hsl(var(--primary)/0.5)]"
        />

        {/* User stats points */}
        {userStats.map((value, i) => {
          const point = getPoint(value, i);
          return (
            <motion.circle
              key={i}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              cx={point.x}
              cy={point.y}
              r="5"
              fill="hsl(var(--primary))"
              stroke="hsl(var(--background))"
              strokeWidth="2"
              className="drop-shadow-[0_0_6px_hsl(var(--primary))]"
            />
          );
        })}

        {/* Labels */}
        {labels.map((label, i) => (
          <text
            key={i}
            x={labelPositions[i].x}
            y={labelPositions[i].y}
            textAnchor="middle"
            dominantBaseline="middle"
            className="fill-muted-foreground font-display text-xs font-medium"
          >
            {label}
          </text>
        ))}
      </svg>

      {/* Legend */}
      <div className="mt-4 flex justify-center gap-6 text-xs">
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-primary shadow-neon" />
          <span className="text-muted-foreground">Your Player</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="h-3 w-3 rounded-full border-2 border-dashed border-stadium-gold bg-stadium-gold/20" />
          <span className="text-muted-foreground">Superstar Avg</span>
        </div>
      </div>
    </div>
  );
}
