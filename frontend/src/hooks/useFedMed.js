/**
 * Live FedMed state hook.
 *
 * Owns the single WebSocket connection to the aggregator and keeps the
 * dashboard in sync with it. The snapshot frame hydrates the whole app;
 * subsequent frames patch the relevant slice.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, openMetricsSocket } from "../lib/api";

const EMPTY_NODES = [];
const EMPTY_ROUNDS = [];

function mergeNode(nodes, hospitalId, patch) {
  const index = nodes.findIndex((n) => n.hospital_id === hospitalId);
  if (index === -1) return nodes;
  const next = nodes.slice();
  next[index] = { ...next[index], ...patch };
  return next;
}

export function useFedMed() {
  const [status, setStatus] = useState(null);
  const [nodes, setNodes] = useState(EMPTY_NODES);
  const [rounds, setRounds] = useState(EMPTY_ROUNDS);
  const [privacyEvents, setPrivacyEvents] = useState([]);
  const [securityEvents, setSecurityEvents] = useState([]);
  const [summary, setSummary] = useState(null);
  const [baseline, setBaseline] = useState(null);
  const [segmentation, setSegmentation] = useState(null);
  const [connection, setConnection] = useState("connecting");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const socketRef = useRef(null);

  const applySnapshot = useCallback((data) => {
    if (data.status) setStatus(data.status);
    if (data.nodes) setNodes(data.nodes);
    if (data.rounds) setRounds(data.rounds);
    if (data.summary) setSummary(data.summary);
    if (data.baseline) setBaseline(data.baseline);
    if (data.privacy_events) setPrivacyEvents(data.privacy_events);
    if (data.security_events) setSecurityEvents(data.security_events);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [statusData, nodesData, metricsData, privacyData] = await Promise.all([
        api.status(),
        api.nodes(),
        api.metrics(),
        api.privacy(),
      ]);

      setStatus(statusData);
      setNodes(nodesData.nodes ?? EMPTY_NODES);
      setRounds(metricsData.rounds ?? EMPTY_ROUNDS);
      setSummary(metricsData.summary ?? null);
      setBaseline(metricsData.baseline ?? null);
      setPrivacyEvents(privacyData.privacy_events ?? []);
      setSecurityEvents(privacyData.security_events ?? []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Hydrate from the REST API, then keep the view live over the socket.
    let cancelled = false;

    Promise.all([api.status(), api.nodes(), api.metrics(), api.privacy()])
      .then(([statusData, nodesData, metricsData, privacyData]) => {
        if (cancelled) return;
        applySnapshot({
          status: statusData,
          nodes: nodesData.nodes ?? EMPTY_NODES,
          rounds: metricsData.rounds ?? EMPTY_ROUNDS,
          summary: metricsData.summary ?? null,
          baseline: metricsData.baseline ?? null,
          privacy_events: privacyData.privacy_events ?? [],
          security_events: privacyData.security_events ?? [],
        });
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    const handleEvent = (event) => {
      const data = event.data || {};

      switch (event.type) {
        case "snapshot": {
          if (data.status) setStatus(data.status);
          if (data.nodes) setNodes(data.nodes);
          if (data.rounds) setRounds(data.rounds);
          if (data.privacy_events) setPrivacyEvents(data.privacy_events);
          if (data.security_events) setSecurityEvents(data.security_events);
          if (data.summary) setSummary(data.summary);
          if (data.baseline) setBaseline(data.baseline);
          if (data.segmentation) setSegmentation(data.segmentation);
          break;
        }
        case "round": {
          const record = data;
          if (!record) break;
          setRounds((prev) => {
            const index = prev.findIndex((r) => r.round === record.round);
            if (index === -1) return [...prev, record];
            const next = prev.slice();
            next[index] = record;
            return next;
          });
          setStatus((prev) =>
            prev
              ? {
                  ...prev,
                  current_round: record.round,
                  is_running: true,
                  dp_epsilon: record.dp_epsilon ?? prev.dp_epsilon,
                }
              : prev,
          );
          break;
        }
        case "node_status": {
          if (!data.hospital_id) break;
          setNodes((prev) =>
            mergeNode(prev, data.hospital_id, {
              status: data.status,
              message: data.message ?? data.status,
            }),
          );
          break;
        }
        case "privacy_event": {
          if (data) setPrivacyEvents((prev) => [...prev.slice(-99), data]);
          break;
        }
        case "security_event": {
          if (data) setSecurityEvents((prev) => [...prev.slice(-99), data]);
          break;
        }
        case "baseline": {
          if (data) setBaseline(data);
          break;
        }
        case "run_started": {
          setStatus(data);
          break;
        }
        case "run_finished": {
          setStatus((prev) => (prev ? { ...prev, is_running: false } : prev));
          refresh();
          break;
        }
        case "segmentation": {
          if (data) setSegmentation(data);
          break;
        }
        case "error": {
          setError(data.error ?? "Unknown error");
          break;
        }
        default:
          break;
      }
    };

    socketRef.current = openMetricsSocket({
      onEvent: handleEvent,
      onStatus: setConnection,
    });

    return () => {
      cancelled = true;
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [applySnapshot, refresh]);

  const derived = useMemo(() => {
    const latest = rounds.length ? rounds[rounds.length - 1] : null;
    const baselineDice = Number(baseline?.best_dice ?? 0);
    const baselineTestDice = Number(baseline?.test_dice ?? 0);
    const federatedDice = Number(latest?.dice ?? 0);
    const federatedValDice = Number(summary?.val_dice ?? summary?.federated_best_dice ?? federatedDice);
    const federatedTestDice = Number(status?.test_dice ?? summary?.test_dice ?? summary?.federated_test_dice ?? 0);
    const onlineNodes = nodes.filter((n) =>
      ["online", "training", "aggregating"].includes(n.status),
    ).length;

    return {
      latest,
      baselineDice,
      baselineTestDice,
      federatedDice,
      federatedValDice,
      federatedTestDice,
      testDice: federatedTestDice,
      valDice: federatedValDice,
      onlineNodes,
      totalNodes: nodes.length || 3,
      diceGap: baselineDice ? Number((baselineDice - federatedDice).toFixed(2)) : null,
      testDiceGap: baselineTestDice && federatedTestDice ? Number((baselineTestDice - federatedTestDice).toFixed(2)) : null,
      diceRatio: baselineDice
        ? Number(((federatedDice / baselineDice) * 100).toFixed(1))
        : null,
      progress: status?.total_rounds
        ? Math.min(100, Math.round((rounds.length / status.total_rounds) * 100))
        : 0,
    };
  }, [rounds, nodes, status, baseline]);

  const actions = useMemo(
    () => ({
      refresh,
      startBaseline: (epochs) => api.startBaseline(epochs),
      startFederated: (roundsCount, dropoutRound) =>
        api.startFederated(roundsCount, dropoutRound),
      analyze: (model, sample) => api.mriAnalyze(model, sample),
      updateSettings: (payload) => api.updateSettings(payload),
    }),
    [refresh],
  );

  return {
    status,
    nodes,
    rounds,
    privacyEvents,
    securityEvents,
    summary,
    baseline,
    segmentation,
    connection,
    error,
    loading,
    ...derived,
    actions,
  };
}
