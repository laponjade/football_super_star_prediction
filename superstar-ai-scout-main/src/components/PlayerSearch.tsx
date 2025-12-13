import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Search, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  searchPlayers as searchPlayersAPI,
  type PlayerSearchResult,
} from "@/lib/api";
import { useNavigate } from "react-router-dom";

export function PlayerSearch() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PlayerSearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const trimmedQuery = query.trim();

    // Require at least 3 characters before searching
    if (!trimmedQuery || trimmedQuery.length < 3) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    setIsSearching(true);
    const timer = setTimeout(async () => {
      try {
        console.log("Searching for:", trimmedQuery);
        const searchResults = await searchPlayersAPI(trimmedQuery, 10);
        console.log("Search results:", searchResults);
        setResults(searchResults);
        setIsOpen(true);
      } catch (error) {
        console.error("Search error:", error);
        setResults([]);
        // Show error to user (optional - you can add a toast here)
      } finally {
        setIsSearching(false);
      }
    }, 300); // Debounce 300ms

    return () => clearTimeout(timer);
  }, [query]);

  const handleSelect = (player: PlayerSearchResult) => {
    // Navigate to predict page with player_id
    navigate(`/predict?player_id=${player.player_id}`);
    setQuery("");
    setIsOpen(false);
  };

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <div className="relative">
        <Search className="absolute left-4 top-1/2 h-5 w-5 -translate-y-1/2 text-muted-foreground" />
        <Input
          ref={inputRef}
          type="text"
          placeholder="Type at least 3 characters to search..."
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
          }}
          onFocus={() => setIsOpen(true)}
          className="h-14 w-full rounded-full border-2 border-primary/30 bg-card/80 pl-12 pr-12 font-body text-lg placeholder:text-muted-foreground/60 focus:border-primary focus:ring-primary/20 focus:shadow-neon backdrop-blur-sm transition-all"
        />
        {isSearching && (
          <Loader2 className="absolute right-4 top-1/2 h-5 w-5 -translate-y-1/2 animate-spin text-primary" />
        )}
        {!isSearching && query && query.length < 3 && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
            Type at least 3 characters
          </div>
        )}
        {!isSearching && query && query.length >= 3 && results.length === 0 && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-muted-foreground">
            No results
          </div>
        )}
      </div>

      <AnimatePresence>
        {isOpen && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute top-full left-0 right-0 z-50 mt-2 overflow-hidden rounded-xl border border-border bg-card/95 shadow-card backdrop-blur-md max-h-96 overflow-y-auto"
          >
            {results.map((player, index) => (
              <motion.button
                key={player.player_id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                onClick={() => handleSelect(player)}
                className="flex w-full items-center gap-4 px-4 py-3 text-left transition-colors hover:bg-primary/10"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/20 font-display text-sm font-bold text-primary">
                  {player.overall}
                </div>
                <div className="flex-1">
                  <p className="font-medium text-foreground">{player.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {player.position} • {player.club} • {player.nationality} •{" "}
                    {player.age} years
                  </p>
                </div>
                <div className="text-right">
                  <span className="text-xs text-primary">
                    POT {player.potential}
                  </span>
                </div>
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
