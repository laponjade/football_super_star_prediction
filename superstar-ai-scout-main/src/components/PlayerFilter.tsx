import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FilteredPlayerResult, filterPlayers } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const positions = [
  { value: "any", label: "Any" },
  { value: "ST", label: "ST" },
  { value: "CF", label: "CF" },
  { value: "LW", label: "LW" },
  { value: "RW", label: "RW" },
  { value: "CAM", label: "CAM" },
  { value: "CM", label: "CM" },
  { value: "CDM", label: "CDM" },
  { value: "LB", label: "LB" },
  { value: "RB", label: "RB" },
  { value: "CB", label: "CB" },
  { value: "GK", label: "GK" },
];

const sortOptions = [
  { value: "superstar_first", label: "Superstars first" },
  { value: "non_superstar_first", label: "Non-superstars first" },
  { value: "overall_desc", label: "Overall (high → low)" },
  { value: "age_asc", label: "Age (young → old)" },
];

// Static list of common nationalities from the FIFA dataset to avoid typos.
// You can extend this list over time if you need more.
const countriesList = [
  "Algeria",
  "Argentina",
  "Australia",
  "Austria",
  "Belgium",
  "Bosnia and Herzegovina",
  "Brazil",
  "Cameroon",
  "Canada",
  "Chile",
  "China PR",
  "Colombia",
  "Croatia",
  "Czech Republic",
  "Denmark",
  "Egypt",
  "England",
  "Finland",
  "France",
  "Germany",
  "Ghana",
  "Greece",
  "Hungary",
  "Iceland",
  "Iran",
  "Ireland",
  "Italy",
  "Ivory Coast",
  "Japan",
  "Mexico",
  "Morocco",
  "Netherlands",
  "New Zealand",
  "Nigeria",
  "Norway",
  "Poland",
  "Portugal",
  "Romania",
  "Russia",
  "Saudi Arabia",
  "Scotland",
  "Senegal",
  "Serbia",
  "Slovakia",
  "Slovenia",
  "South Africa",
  "South Korea",
  "Spain",
  "Sweden",
  "Switzerland",
  "Tunisia",
  "Turkey",
  "Ukraine",
  "United States",
  "Uruguay",
  "Venezuela",
  "Wales",
];

export function PlayerFilter() {
  const navigate = useNavigate();
  const [country, setCountry] = useState("");
  const [position, setPosition] = useState<string>("any");
  const [minAge, setMinAge] = useState<number | undefined>(18);
  const [maxAge, setMaxAge] = useState<number | undefined>(30);
  const [sortBy, setSortBy] = useState<string>("superstar_first");
  const [isLoading, setIsLoading] = useState(false);
  const [countrySearch, setCountrySearch] = useState("");
  const [results, setResults] = useState<FilteredPlayerResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFilter = async () => {
    setError(null);

    if (!country.trim()) {
      setError("Please enter a country to filter.");
      return;
    }

    setIsLoading(true);
    try {
      const data = await filterPlayers(country, {
        position: position === "any" ? undefined : position,
        minAge,
        maxAge,
        limit: 50,
      });
      setResults(data);
    } catch (e: any) {
      console.error("Filter error:", e);
      setError(e.message || "Failed to filter players. Please try again.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  const sortedResults = useMemo(() => {
    const items = [...results];

    switch (sortBy) {
      case "superstar_first":
        return items.sort((a, b) => {
          if (a.will_be_superstar === b.will_be_superstar) {
            return b.overall - a.overall;
          }
          return a.will_be_superstar ? -1 : 1;
        });
      case "non_superstar_first":
        return items.sort((a, b) => {
          if (a.will_be_superstar === b.will_be_superstar) {
            return b.overall - a.overall;
          }
          return a.will_be_superstar ? 1 : -1;
        });
      case "overall_desc":
        return items.sort((a, b) => b.overall - a.overall);
      case "age_asc":
        return items.sort((a, b) => a.age - b.age);
      default:
        return items;
    }
  }, [results, sortBy]);

  const filteredCountries = useMemo(() => {
    if (!countrySearch.trim()) return countriesList;
    const q = countrySearch.toLowerCase();
    return countriesList.filter((c) => c.toLowerCase().includes(q));
  }, [countrySearch]);

  return (
    <section className="mt-16">
      <div className="mb-6 text-center">
        <h3 className="font-display text-xl font-semibold text-foreground">
          Explore FIFA 21 Prospects
        </h3>
        <p className="text-sm text-muted-foreground">
          Filter FIFA 21 players by country, position and age, and see who the
          model thinks will be a future superstar.
        </p>
      </div>

      <div className="mx-auto flex max-w-3xl flex-col gap-4 rounded-xl border border-border bg-card/60 p-4 backdrop-blur-sm md:flex-row md:items-end">
        <div className="flex-1">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            Country (nationality)
          </label>
          <Select value={country} onValueChange={(value) => setCountry(value)}>
            <SelectTrigger className="bg-input border-border">
              <SelectValue placeholder="Select a country" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border max-h-72">
              <div className="px-2 py-1.5">
                <Input
                  placeholder="Search country..."
                  value={countrySearch}
                  onChange={(e) => setCountrySearch(e.target.value)}
                  className="h-8 text-xs"
                />
              </div>
              {filteredCountries.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="w-full md:w-48">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            Position (optional)
          </label>
          <Select
            value={position}
            onValueChange={(value) => setPosition(value)}
          >
            <SelectTrigger className="bg-input border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {positions.map((pos) => (
                <SelectItem key={pos.value} value={pos.value}>
                  {pos.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex w-full gap-2 md:w-56">
          <div className="flex-1">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              Min Age
            </label>
            <Input
              type="number"
              min={16}
              max={45}
              value={minAge ?? ""}
              onChange={(e) =>
                setMinAge(e.target.value ? Number(e.target.value) : undefined)
              }
            />
          </div>
          <div className="flex-1">
            <label className="mb-1 block text-xs font-medium text-muted-foreground">
              Max Age
            </label>
            <Input
              type="number"
              min={16}
              max={45}
              value={maxAge ?? ""}
              onChange={(e) =>
                setMaxAge(e.target.value ? Number(e.target.value) : undefined)
              }
            />
          </div>
        </div>

        <div className="w-full md:w-56">
          <label className="mb-1 block text-xs font-medium text-muted-foreground">
            Sort by
          </label>
          <Select value={sortBy} onValueChange={(value) => setSortBy(value)}>
            <SelectTrigger className="bg-input border-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {sortOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <Button
          onClick={handleFilter}
          disabled={isLoading}
          className="w-full md:w-auto rounded-full"
        >
          {isLoading ? "Scanning..." : "Filter Players"}
        </Button>
      </div>

      {error && (
        <div className="mx-auto mt-3 max-w-3xl text-center text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Results */}
      <div className="mx-auto mt-6 max-w-4xl">
        {sortedResults.length > 0 ? (
          <div className="grid gap-4 md:grid-cols-2">
            {sortedResults.map((player) => (
              <div
                key={player.player_id}
                className="cursor-pointer rounded-xl border border-border bg-card/70 p-4 transition-colors hover:border-primary/60 hover:bg-card"
                onClick={() => navigate(`/player/${player.player_id}`)}
              >
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <h4 className="font-display text-sm font-semibold text-foreground">
                      {player.name}
                    </h4>
                    <p className="text-xs text-muted-foreground">
                      {player.position} • {player.club} • {player.nationality}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="rounded-full bg-primary/15 px-3 py-1 text-xs font-medium text-primary">
                      OVR {player.overall} • POT {player.potential}
                    </div>
                  </div>
                </div>

                <div className="mt-3 flex items-center justify-between gap-2">
                  <div className="text-xs text-muted-foreground">
                    Age {player.age} • Pace {player.pace ?? "-"} • Dribbling{" "}
                    {player.dribbling ?? "-"}
                  </div>
                  <div
                    className={`rounded-full px-3 py-1 text-xs font-semibold ${
                      player.will_be_superstar
                        ? "bg-emerald-500/15 text-emerald-400"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {player.superstar_label}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          !isLoading && (
            <p className="mt-4 text-center text-sm text-muted-foreground">
              No players loaded yet. Choose a country and hit{" "}
              <span className="font-semibold text-primary">Filter Players</span>{" "}
              to explore the FIFA 21 database.
            </p>
          )
        )}
      </div>
    </section>
  );
}

