import { motion, AnimatePresence } from "framer-motion";
import { Loader2 } from "lucide-react";

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
}

export function LoadingOverlay({ isLoading, message = "Analyzing DNA..." }: LoadingOverlayProps) {
  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/90 backdrop-blur-sm"
        >
          <div className="flex flex-col items-center gap-6">
            {/* Floodlight animation */}
            <div className="relative h-32 w-32">
              <motion.div
                className="absolute inset-0 rounded-full bg-primary/20"
                animate={{ 
                  scale: [1, 1.5, 1],
                  opacity: [0.5, 0.2, 0.5]
                }}
                transition={{ duration: 2, repeat: Infinity }}
              />
              <motion.div
                className="absolute inset-4 rounded-full bg-primary/40"
                animate={{ 
                  scale: [1, 1.3, 1],
                  opacity: [0.7, 0.3, 0.7]
                }}
                transition={{ duration: 2, repeat: Infinity, delay: 0.3 }}
              />
              <div className="absolute inset-8 flex items-center justify-center rounded-full bg-primary/60">
                <Loader2 className="h-8 w-8 animate-spin text-primary-foreground" />
              </div>
            </div>

            <motion.p
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="font-display text-lg font-medium text-primary text-glow"
            >
              {message}
            </motion.p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
