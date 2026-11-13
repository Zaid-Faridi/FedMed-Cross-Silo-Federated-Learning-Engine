import { useState } from "react";

import ConnectionBadge from "./components/ConnectionBadge";
import StatCard from "./components/StatCard";
import TrainingChart from "./components/TrainingChart";
import { useFedMed } from "./hooks/useFedMed";

import "./App.css";

import Hospitals from "./pages/Hospitals";
import Training from "./pages/Training";
import MRIAnalysis from "./pages/MRIAnalysis";
import Privacy from "./pages/Privacy";
import Settings from "./pages/Settings";

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard" },
  { id: "hospitals", label: "Hospitals" },
  { id: "training", label: "Training" },
  { id: "mri", label: "MRI Analysis" },
  { id: "privacy", label: "Privacy & Security" },
];

const PAGES = {
  hospitals: Hospitals,
  training: Training,
  mri: MRIAnalysis,
  privacy: Privacy,
  settings: Settings,
};

/**
 * The sidebar is declared at module scope (not inside the component) so
 * React does not remount it on every render.
 */
function Sidebar({ currentPage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="logo">
        <span>🏥</span>
        <h2>FedMed</h2>
      </div>

      <nav>
        {NAV_ITEMS.map((item) => (
          <a
            key={item.id}
            className={currentPage === item.id ? "active" : ""}
            onClick={() => onNavigate(item.id)}
            href="#"
          >
            {item.label}
          </a>
        ))}
      </nav>

      <div className="sidebar-bottom">
        <a
          className={currentPage === "settings" ? "active" : ""}
          onClick={() => onNavigate("settings")}
          href="#"
        >
          ⚙️ Settings
        </a>
      </div>
    </aside>
  );
}

function Dashboard({ fed, onNavigate }) {
  const {
    status,
    nodes,
    rounds,
    latest,
    baselineDice,
    baselineTestDice,
    federatedDice,
    federatedValDice,
    federatedTestDice,
    onlineNodes,
    totalNodes,
    diceGap,
    diceRatio,
    progress,
    connection,
    loading,
    segmentation,
  } = fed;

  const chartData = rounds.map((r) => ({
    round: r.round,
    loss: Number(r.loss ?? 0),
    dice: Number(r.dice ?? 0),
  }));

  const roundCaption =
    status?.current_round && status?.total_rounds
      ? `${status.current_round} of ${status.total_rounds} rounds complete`
      : "No run started yet";

  const diceCaption =
    baselineDice && diceGap !== null
      ? `${diceRatio}% of baseline val (${diceGap}% gap)`
      : "Run a baseline to enable the comparison";

  const testDiceCaption =
    baselineTestDice && federatedTestDice
      ? `Baseline test: ${baselineTestDice.toFixed(1)}%`
      : "Evaluated on held-out test split";

  const lossCaption = latest
    ? `Round ${latest.round} · ${latest.aggregation ?? "aggregation"}`
    : "Awaiting the first round";

  return (
    <main className="main-content">
      <header className="topbar">
        <div>
          <h1>Federated Learning Dashboard</h1>
          <p>Privacy-preserving healthcare AI</p>
        </div>

        <div className="topbar-right">
          {fed.error ? <div className="topbar-error">{fed.error}</div> : null}
          <ConnectionBadge
            connection={connection}
            running={Boolean(status?.is_running)}
          />
        </div>
      </header>

      <section className="stats">
        <StatCard
          label="Training Round"
          value={status?.current_round ?? 0}
          caption={roundCaption}
          icon="🔁"
        />
        <StatCard
          label="Validation Dice"
          value={latest ? `${(federatedValDice || federatedDice).toFixed(2)}%` : "—"}
          caption={diceCaption}
          tone="success"
          icon="🎯"
        />
        <StatCard
          label="Held-Out Test Dice"
          value={federatedTestDice ? `${federatedTestDice.toFixed(2)}%` : "—"}
          caption={testDiceCaption}
          tone="info"
          icon="🧪"
        />
        <StatCard
          label="Active Hospitals"
          value={`${onlineNodes} / ${totalNodes}`}
          caption={`${progress}% of rounds completed`}
          tone="info"
          icon="🏥"
        />
      </section>

      <section className="content-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>Training Progress</h3>
              <p>Global model performance across federated rounds</p>
            </div>
            <span className="badge">{connection === "online" ? "Live" : "Paused"}</span>
          </div>

          <TrainingChart data={chartData} baselineDice={baselineDice} />
        </div>

        <div className="panel">
          <div className="panel-header">
            <div>
              <h3>Hospital Nodes</h3>
              <p>Federated clients</p>
            </div>
          </div>

          <div className="hospital-list">
            {loading && !nodes.length ? (
              <p className="muted">Loading node status…</p>
            ) : null}

            {nodes.map((node) => (
              <div className="hospital" key={node.hospital_id}>
                <div className="hospital-icon">🏥</div>
                <div>
                  <strong>
                    {node.hospital_id?.replace("hospital-", "Hospital ")}
                  </strong>
                  <p>{node.message ?? "Awaiting connection"}</p>
                </div>
                <span
                  className={
                    ["online", "training", "aggregating"].includes(node.status)
                      ? "online"
                      : "offline"
                  }
                >
                  ●
                </span>
              </div>
            ))}
          </div>

          {baselineDice ? (
            <div className="baseline-callout">
              <span>Centralized baseline</span>
              <strong>{baselineDice.toFixed(2)}% Dice</strong>
              <p>
                The federated global model reaches{" "}
                {diceRatio ?? 0}% of the centralized accuracy without any raw
                patient data leaving the hospitals.
              </p>
            </div>
          ) : null}
        </div>
      </section>

      <section className="panel segmentation">
        <div className="panel-header">
          <div>
            <h3>🧠 MRI Tumor Segmentation</h3>
            <p>Latest model prediction</p>
          </div>
          <button onClick={() => onNavigate("mri")}>View Analysis</button>
        </div>

        <div className="mri-section">
          <div className="mri-box">
            {segmentation?.mri_png ? (
              <img
                className="mri-thumb"
                src={`data:image/png;base64,${segmentation.mri_png}`}
                alt="MRI slice"
              />
            ) : (
              <div>🧠</div>
            )}
            <p>MRI Scan</p>
          </div>
          <div className="arrow">→</div>
          <div className="mri-box">
            {segmentation?.mask_png ? (
              <img
                className="mri-thumb"
                src={`data:image/png;base64,${segmentation.mask_png}`}
                alt="Tumor mask overlay"
              />
            ) : (
              <div>🎯</div>
            )}
            <p>Tumor Mask</p>
          </div>
          <div className="result">
            <span>Segmentation Accuracy</span>
            <strong>
              {segmentation?.volume_dice !== undefined
                ? `${segmentation.volume_dice}%`
                : `${federatedDice.toFixed(2)}%`}
            </strong>
          </div>
        </div>
      </section>

      <footer>
        <span>FedMed • Federated Healthcare AI</span>
        <span>🔐 Privacy Protected</span>
      </footer>
    </main>
  );
}

function App() {
  const [currentPage, setCurrentPage] = useState("dashboard");
  const fed = useFedMed();

  const ActivePage = PAGES[currentPage];

  return (
    <div className="app">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />

      {ActivePage ? (
        <ActivePage fed={fed} onNavigate={setCurrentPage} />
      ) : (
        <Dashboard fed={fed} onNavigate={setCurrentPage} />
      )}
    </div>
  );
}

export default App;
