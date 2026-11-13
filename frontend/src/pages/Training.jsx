import { useState } from "react";

import StatCard from "../components/StatCard";
import TrainingChart from "../components/TrainingChart";

const STATUS_LABEL = {
  online: "Online",
  training: "Training",
  aggregating: "Encrypting",
  connecting: "Connecting",
  offline: "Offline",
  failed: "Failed",
};

/**
 * Federated training monitor.
 *
 * Lets an operator start a baseline run or a federated run (optionally
 * with a simulated node dropout) and watch the rounds stream in live.
 */
function Training({ fed }) {
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
    progress,
    actions,
    connection,
  } = fed;

  const [roundsInput, setRoundsInput] = useState(10);
  const [dropoutRound, setDropoutRound] = useState(0);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);

  const running = Boolean(status?.is_running);

  const handleStart = async () => {
    setBusy(true);
    setNotice(null);
    try {
      await actions.startFederated(Number(roundsInput), Number(dropoutRound));
      setNotice("Federated training started");
    } catch (err) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  };

  const handleBaseline = async () => {
    setBusy(true);
    setNotice(null);
    try {
      await actions.startBaseline(30);
      setNotice("Centralized baseline training started");
    } catch (err) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  };

  const chartData = rounds.map((r) => ({
    round: r.round,
    loss: Number(r.loss ?? 0),
    dice: Number(r.dice ?? 0),
  }));

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Training</h1>
          <p>Monitor federated model training progress</p>
        </div>

        <div className="training-status">
          <span className="training-dot" />
          {running ? "Training Active" : "Idle"}
        </div>
      </div>

      <div className="training-stats">
        <StatCard
          label="Current Round"
          value={`${status?.current_round ?? 0} / ${status?.total_rounds ?? 0}`}
          caption={`${progress}% completed`}
        />
        <StatCard
          label="Validation Dice"
          value={latest ? `${(federatedValDice || federatedDice).toFixed(2)}%` : "—"}
          caption={
            baselineDice
              ? `${(( (federatedValDice || federatedDice) / baselineDice) * 100).toFixed(1)}% of baseline val (${baselineDice.toFixed(1)}%)`
              : "Best round on val split"
          }
          tone="success"
        />
        <StatCard
          label="Held-Out Test Dice"
          value={federatedTestDice ? `${federatedTestDice.toFixed(2)}%` : "—"}
          caption={
            baselineTestDice
              ? `Baseline test: ${baselineTestDice.toFixed(1)}%`
              : "Evaluated on held-out test split"
          }
          tone="info"
        />
        <StatCard
          label="Current Loss"
          value={latest ? Number(latest.loss ?? 0).toFixed(4) : "—"}
          caption={latest ? `Round ${latest.round}` : "Awaiting first round"}
          tone="danger"
        />
        <StatCard
          label="Participants"
          value={`${nodes.filter((n) => n.rounds_participated > 0).length} / ${
            nodes.length || 3
          }`}
          caption="Hospitals that joined a round"
          tone="info"
        />
      </div>

      <div className="panel training-progress-panel">
        <div className="panel-header">
          <div>
            <h3>Federated Training Progress</h3>
            <p>Current global model training status</p>
          </div>
          <span className="badge">
            {status?.current_round ? `Round ${status.current_round}` : "Idle"}
          </span>
        </div>

        <div className="progress-container">
          <div className="progress-info">
            <span>Overall Progress</span>
            <strong>{progress}%</strong>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>

        <div className="training-process">
          {[
            { label: "Model Broadcast", note: "Global model sent", done: rounds.length > 0 },
            { label: "Local Training", note: "Hospitals training", done: rounds.length > 0 },
            {
              label: "Aggregation",
              note: latest?.aggregation ?? "FedAvg",
              done: Boolean(latest),
            },
            {
              label: "Global Update",
              note: running ? "In progress" : "Waiting",
              done: false,
            },
          ].map((step, index, all) => (
            <div className="process-step-wrap" key={step.label}>
              <div className={`process-step ${step.done ? "completed" : ""}`}>
                <div className="process-icon">{step.done ? "✓" : index + 1}</div>
                <strong>{step.label}</strong>
                <span>{step.note}</span>
              </div>
              {index < all.length - 1 ? <div className="process-line" /> : null}
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Launch a Run</h3>
            <p>Start the centralized baseline or the federated loop</p>
          </div>
        </div>

        <div className="launch-controls">
          <div className="form-group">
            <label htmlFor="rounds">Federated Rounds</label>
            <input
              id="rounds"
              type="number"
              min="1"
              max="100"
              value={roundsInput}
              onChange={(e) => setRoundsInput(e.target.value)}
            />
            <small>Number of global training rounds</small>
          </div>

          <div className="form-group">
            <label htmlFor="dropout">Node Dropout Round</label>
            <input
              id="dropout"
              type="number"
              min="0"
              max="100"
              value={dropoutRound}
              onChange={(e) => setDropoutRound(e.target.value)}
            />
            <small>0 disables the resilience test</small>
          </div>

          <div className="launch-buttons">
            <button
              className="save-button"
              onClick={handleStart}
              disabled={busy || running}
            >
              {running ? "Running…" : "Start Federated Run"}
            </button>
            <button
              className="cancel-button"
              onClick={handleBaseline}
              disabled={busy || running}
            >
              Run Centralized Baseline
            </button>
          </div>

          {notice ? <p className="launch-notice">{notice}</p> : null}
          <p className="muted">
            WebSocket: {connection}. The aggregator aggregates encrypted
            ciphertexts and decrypts only the aggregate.
          </p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-header">
          <div>
            <h3>Hospital Training Status</h3>
            <p>Local training progress for each node</p>
          </div>
        </div>

        <div className="training-hospitals">
          {nodes.map((node) => {
            const pct = Math.min(
              100,
              ((node.rounds_participated ?? 0) / (status?.total_rounds || 1)) * 100,
            );
            return (
              <div className="training-hospital" key={node.hospital_id}>
                <div className="training-hospital-left">
                  <div className="hospital-icon">🏥</div>
                  <div>
                    <strong>
                      {node.hospital_id?.replace("hospital-", "Hospital ")}
                    </strong>
                    <p>
                      {STATUS_LABEL[node.status] ?? node.status} ·{" "}
                      {node.rounds_participated ?? 0} rounds
                    </p>
                  </div>
                </div>

                <div className="hospital-progress">
                  <div className="mini-progress">
                    <div
                      className="mini-progress-fill"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span>{Math.round(pct)}%</span>
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
            );
          })}
        </div>
      </div>

      <div className="panel training-history">
        <div className="panel-header">
          <div>
            <h3>Training History</h3>
            <p>Federated learning round statistics</p>
          </div>
        </div>

        <div className="chart">
          <TrainingChart data={chartData} baselineDice={baselineDice} height={260} />
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Round</th>
                <th>Training Loss</th>
                <th>Dice Score</th>
                <th>Aggregation</th>
                <th>Encrypted</th>
                <th>ε (DP)</th>
                <th>Status</th>
              </tr>
            </thead>

            <tbody>
              {rounds.length === 0 ? (
                <tr>
                  <td colSpan="7" className="muted">
                    No rounds recorded yet.
                  </td>
                </tr>
              ) : null}

              {[...rounds].reverse().map((item) => (
                <tr key={item.round}>
                  <td>
                    <strong>Round {item.round}</strong>
                  </td>
                  <td>{Number(item.loss ?? 0).toFixed(4)}</td>
                  <td>
                    <strong>{Number(item.dice ?? 0).toFixed(2)}%</strong>
                  </td>
                  <td>{item.aggregation}</td>
                  <td>{item.encrypted_updates ?? 0} / {item.clients ?? 0}</td>
                  <td>{Number(item.dp_epsilon ?? 0).toFixed(3)}</td>
                  <td>
                    <span
                      className={
                        item.failed_clients
                          ? "round-running"
                          : "round-completed"
                      }
                    >
                      ● {item.failed_clients ? "Degraded" : "Completed"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default Training;
