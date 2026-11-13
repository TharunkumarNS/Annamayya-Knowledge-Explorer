
import {
  House,
  Search,
  Music2,
  AudioLines,
  Sparkles,
  Network,
  Library,
  Heart,
  Settings,
  Plus,
} from "lucide-react";

const menuItems = [
  { name: "Overview", icon: House },
  { name: "Discover Songs", icon: Search },
  { name: "Audio Search", icon: AudioLines },
  { name: "AI Companion", icon: Sparkles },
  { name: "Knowledge Graph", icon: Network },
  { name: "Song Library", icon: Library },
  { name: "Favourites", icon: Heart },
];

type SidebarProps = {
  activePage: string;
  onNavigate: (page: string) => void;
};

export default function Sidebar({
  activePage,
  onNavigate,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <Music2 size={23} />
        </div>
        <div>
          <h2>Annamayya</h2>
          <span>KNOWLEDGE EXPLORER</span>
        </div>
      </div>

      <button
        className="new-exploration"
        onClick={() => onNavigate("Overview")}
      >
        <Plus size={17} />
        New exploration
      </button>

      <p className="menu-heading">WORKSPACE</p>

      <nav className="sidebar-menu">
        {menuItems.map((item) => {
          const Icon = item.icon;

          return (
            <button
              key={item.name}
              className={
                activePage === item.name
                  ? "sidebar-item active"
                  : "sidebar-item"
              }
              onClick={() => onNavigate(item.name)}
            >
              <Icon size={18} />
              <span>{item.name}</span>
            </button>
          );
        })}
      </nav>

      <div className="sidebar-bottom">
        <button className="sidebar-item">
          <Settings size={18} />
          <span>Settings</span>
        </button>
        <p>Preserving musical heritage.</p>
      </div>
    </aside>
  );
}
