
import { useState } from "react";

import {
  Search,
  AudioLines,
  Sparkles,
  Network,
  BookOpen,
  Heart,
  ArrowUpRight,
} from "lucide-react";

import ParticleSphere, {
  type SphereMode,
} from "./ParticleSphere";

// Features available in Annamayya Explorer
const features = [
  {
    title: "Discover Songs",
    description: "Find compositions through Telugu lyrics, titles and meanings.",
    icon: Search,
    color: "blue",
  },
  {
    title: "Audio Search",
    description: "Upload a recording and discover similar compositions.",
    icon: AudioLines,
    color: "purple",
  },
  {
    title: "AI Companion",
    description: "Ask questions and understand the stories behind songs.",
    icon: Sparkles,
    color: "gold",
  },
  {
    title: "Knowledge Graph",
    description: "Explore connections between songs, ragas and singers.",
    icon: Network,
    color: "green",
  },
  {
    title: "Lyrics & Meanings",
    description: "Read Telugu lyrics, transliterations and explanations.",
    icon: BookOpen,
    color: "pink",
  },
  {
    title: "My Favourites",
    description: "Keep your favourite compositions in one place.",
    icon: Heart,
    color: "red",
  },
];

type DashboardProps = {
  onNavigate: (page: string) => void;
};

export default function Dashboard({
  onNavigate,
}: DashboardProps) {

  // Controls the particle animation
  const [sphereMode, setSphereMode] =
    useState<SphereMode>("idle");

  // Stores the search text
  const [search, setSearch] = useState("");

  // Handles search interaction
  const handleSearch = (value: string) => {
    setSearch(value);

    if (value.trim().length > 0) {
      setSphereMode("searching");
    } else {
      setSphereMode("idle");
    }
  };

  return (
    <main className="dashboard">

      {/* Dashboard heading */}
      <header className="dashboard-header">
        <div>
          <p className="dashboard-date">
            ANNAMAYYA KNOWLEDGE EXPLORER
          </p>

          <h1>Your musical universe</h1>

          <p className="dashboard-subtitle">
            Discover compositions, meanings and musical connections.
          </p>
        </div>
      </header>

      {/* Animated particle sphere */}
      <section className="orb-section">

        <div className="orb-glow" />

        <ParticleSphere mode={sphereMode} />

        <h2>Explore the world of Annamayya</h2>

        <p>
          Where music, poetry and technology come together.
        </p>

      </section>

      {/* Search bar */}
      <section className="quick-search">

        <Search size={20} />

        <input
          type="text"
          placeholder="Search songs, lyrics or ask anything..."
          value={search}
          onFocus={() => setSphereMode("searching")}
          onChange={(event) =>
            handleSearch(event.target.value)
          }
        />

        <button
          aria-label="Voice search"
          onClick={() => setSphereMode("searching")}
        >
          <AudioLines size={19} />
        </button>

      </section>

      {/* Animation preview controls */}
      <section className="animation-controls">

        <span>PREVIEW ANIMATION</span>

        {(
          ["idle", "playing", "searching", "connections"] as SphereMode[]
        ).map((mode) => (

          <button
            key={mode}
            className={
              sphereMode === mode
                ? "selected-mode"
                : ""
            }
            onClick={() => setSphereMode(mode)}
          >
            {mode}
          </button>

        ))}

      </section>

      {/* Feature cards */}
      <section className="features-section">

        <div className="features-heading">

          <div>
            <h2>Explore all features</h2>

            <p>
              Everything you need, just one click away.
            </p>
          </div>

          <span>06 FEATURES</span>

        </div>

        <div className="feature-grid">

          {features.map((feature) => {

            const Icon = feature.icon;

            return (

              <button
                className="feature-card"
                key={feature.title}
                onClick={() => {

                  if (feature.title === "Knowledge Graph") {
                    setSphereMode("connections");
                  } else {
                    setSphereMode("idle");
                  }

                  onNavigate(feature.title);
                }}
              >

                <div className={`feature-icon ${feature.color}`}>
                  <Icon size={23} />
                </div>

                <div className="feature-card-text">

                  <h3>{feature.title}</h3>

                  <p>{feature.description}</p>

                </div>

                <ArrowUpRight
                  className="feature-arrow"
                  size={18}
                />

              </button>

            );

          })}

        </div>

      </section>

    </main>
  );
}
