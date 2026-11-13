
import { useState } from "react";
import Sidebar from "./components/Sidebar";
import Dashboard from "./components/Dashboard";
import "./App.css";

function App() {
  const [activePage, setActivePage] = useState("Overview");

  return (
    <div className="app-layout">
      <Sidebar
        activePage={activePage}
        onNavigate={setActivePage}
      />

      <div className="main-content">
        {activePage === "Overview" ? (
          <Dashboard onNavigate={setActivePage} />
        ) : (
          <div className="placeholder-page">
            <h1>{activePage}</h1>
            <p>This section will be developed next.</p>
            <button onClick={() => setActivePage("Overview")}>
              Back to Overview
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
