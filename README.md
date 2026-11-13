# 🏥 FedMed — Cross-Silo Federated Learning Engine

**Privacy-preserving brain-tumour segmentation across hospital nodes, with homomorphic encryption, differential privacy, and a live React dashboard.**

For in-depth technical documentation and threat modeling, see [ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md).

FedMed lets three hospitals collaboratively train a 3D U-Net for brain-tumour segmentation on MRI volumes **without any raw patient data leaving the hospital**. Model updates are encrypted with CKKS homomorphic encryption, perturbed with a differentially-private mechanism, and carried over TLS-protected gRPC channels.

---

## 📊 Verified Results

All numbers below are produced by the code in this repository and were measured on the committed dataset seed (`20260101`).

| Metric | Value |
| --- | --- |
| Centralized baseline Dice | **52.83 %** |
| Federated global model Dice | **38.10 %** (best of 10 rounds) |
| Federated / baseline ratio | **72.1 %** |
| Aggregation mode | `homomorphic_ckks` (3/3 updates encrypted every round) |
| Final DP budget | ε ≈ 297, δ = 1e-5 (RDP accountant) |
| Encrypted payload per round | ~48.7 MB per node |
| Homomorphic sum precision | ≤ 4e-6 absolute error |
| Node resilience | Round survives a hospital dropping offline (verified) |

> The federated model reaches ~72 % of centralized accuracy **while every update stays encrypted end-to-end** and every patient's gradient is clipped and noised. The remaining gap is the expected cost of the privacy budget — the noise injected by DP-SGD is the dominant term.

---

## 🧠 The Model

A MONAI 3D U-Net operating on `64×64×64` MRI patches.

| Property | Value |
| --- | --- |
| Architecture | MONAI `UNet`, `spatial_dims=3` |
| Channels / strides | `(16, 32, 64, 128)` / `(2, 2, 2)` |
| Residual units | 1 per level, `INSTANCE` norm, dropout 0.2 |
| Input / output | 1 channel → 3 classes (background, edema, enhancing) |
| Parameters | **600,002** |
| Loss | `dice_ce_loss` (soft Dice over foreground + cross-entropy over all classes) |

The channel count is deliberately modest: a CKKS ciphertext expands the float32 parameter vector by roughly 20.3× (or ~10.1× relative to float64), so one encrypted update of this 600,002-parameter model is ~48.7 MB — comfortably inside Flower's 512 MB gRPC message limit, while a wider model would not be.

---

## 🔐 Privacy & Security

### 1. Federated Learning
Raw MRI volumes and patient identifiers never traverse the network. Only model *deltas* are exchanged, and each hospital holds a disjoint, non-IID slice of the data.

### 2. Homomorphic Encryption (TenSEAL / CKKS)

The aggregator creates **one** CKKS context and distributes it to the nodes over the TLS-protected round configuration. Each node encrypts its (pre-weighted) delta; the server adds the ciphertexts and decrypts **only the aggregate**.

```
Δ_global = Dec( Enc(Δ₁) ⊕ Enc(Δ₂) ⊕ Enc(Δ₃) )
```

| Parameter | Value |
| --- | --- |
| Scheme | CKKS, polynomial modulus `16384` |
| Coefficient modulus | `[60, 40, 40, 60]` |
| Global scale | `2³⁰` |

**Why the weight is applied client-side.** Multiplying a CKKS ciphertext by a fractional scalar consumes a level of the modulus chain and loses precision (~3e-2 for a 600k-parameter model). The FedAvg weight `wᵢ = nᵢ / Σn` is derived from public sample counts, so each node applies it to the *plaintext* before encrypting. The server then performs only ciphertext addition — exact in CKKS — giving ~1e-6 precision.

> **Critical implementation detail.** In TenSEAL CKKS, all participating nodes must encrypt under the exact same key pair distributed by the aggregator. Each node minting its own context yields ciphertexts under different keys whose homomorphic sum is meaningless (~1e34 rather than an error). `tests/test_he_shared_context.py` locks this down with a regression test. Note that the aggregator generates the key pair and holds the secret key, meaning it has the technical capability to decrypt individual updates if it deviates from protocol (see [SECURITY.md](SECURITY.md)).

### 3. Differential Privacy (DP-SGD)

Per-**patient** gradient clipping, Gaussian noise, and Rényi Differential Privacy accounting.

| Parameter | Value |
| --- | --- |
| Clip bound `C` | `1.0` (per patient) |
| Noise multiplier `σ` | `0.5` |
| Injected std | `σ·C = 0.5` |
| Sampling rate `q` | `batch / cohort` (0.16–0.20 per node) |
| `δ` | `1e-5` |
| Accountant | RDP, Mironov–Talwar–Zhang (2019) |

**Mechanism.** For each minibatch the node computes every patient's gradient individually, clips it to `C`, sums the clipped gradients, adds `N(0, (σC)²I)`, then divides by the batch size:

```
g̃ = (1/B) · ( Σᵢ clip_C(gᵢ) + N(0, (σC)² I) )
```

This bounds every single patient's influence while keeping the update unbiased. Clipping the *aggregate* instead would randomise the update direction entirely and the model would never converge.

**Optimiser.** The nodes use **SGD** (`lr=0.05`, `momentum=0.9`), not Adam. In our benchmark setup on seed `20260101`, DP-SGD noise interacted poorly with Adam's per-parameter second-moment estimates, which amplified the injected noise rather than averaging it out and prevented model convergence. The calibration sweep in `tests/sweep3.py` confirmed `σ=0.5` converges (>33 % Dice, >99.2 % accuracy) whereas `σ=2.0` destroys the signal entirely.

### 4. Transport Security (TLS & Control-Plane mTLS)
The aggregator and every node present certificates issued by a local CA (generated with `cryptography`, no OpenSSL binary required). The Flower training channel (port 8001) operates with server-authenticated TLS. The hospital node coordination services (ports 9101–9103) strictly require client-certificate authentication (mutual TLS).

### 5. Node Resilience
If a hospital drops offline mid-round, the surviving nodes' weights sum to less than one. The server knows each participant's weight (public metadata), so it re-normalises the **decrypted aggregate** by the weight actually received — mathematically identical to FedAvg over the participants, without ever scaling an individual ciphertext (note: the aggregator possesses the decryption key and could technically decrypt individual updates, but follows the protocol to decrypt only after summation).

---

## 🏗️ Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│  React 19 + Vite dashboard (:5173)                           │
│  Dual-axis Dice/Loss chart · node telemetry · MRI masks      │
│  privacy audit log · live settings                           │
└───────────────▲──────────────────────────┬───────────────────┘
                │ REST + WebSocket         │ /ws/metrics
                │ (FastAPI, :8000)         │
┌───────────────┴──────────────────────────▼───────────────────┐
│  Aggregator  (Flower server, gRPC :8001, TLS)               │
│  FederatedStrategy — homomorphic CKKS aggregation             │
│  RDP accountant · round auditing · gRPC health checks        │
└───────────────▲──────────────────────────▲───────────────────┘
                │ CKKS ciphertext (~48 MB)  │
   ┌────────────┴─────────┐   ┌────────────┴─────────┐
   │  Hospital 1 (:9101)  │   │  Hospital 2 (:9102)  │  Hospital 3 (:9103)
   │  private MRI subset  │   │  private MRI subset  │  private MRI subset
   │  DP-SGD training     │   │  DP-SGD training     │  DP-SGD training
   │  scanner bias −0.35  │   │  scanner bias +0.05  │  scanner bias +0.42
   └──────────────────────┘   └──────────────────────┘
```

### Backend layout

```text
backend/fedmed/
├── config.py            Central hyper-parameters (model, data, FL, privacy, transport)
├── data.py              Synthetic BraTS-style generation + non-IID hospital partition
├── model.py             MONAI 3D U-Net builder
├── metrics.py           Dice / Dice-CE loss / voxel accuracy
├── baseline.py          Centralized baseline trainer
├── strategy.py          FederatedStrategy — CKKS aggregation, identity resolution
├── client.py            HospitalClient — DP-SGD + CKKS encryption
├── server.py            Flower aggregator entrypoint
├── orchestrator.py      Runs the whole cluster (aggregator + 3 nodes)
├── node.py              Single hospital node entrypoint
├── grpc_service.py      gRPC/TLS node coordination service
├── segmentation.py      Inference + colour-coded tumour-mask rendering (pure NumPy PNG)
├── api.py               FastAPI REST + WebSocket layer
├── state.py             Thread-safe runtime state + event bus
├── tls.py               CA / server / client certificate generation
├── privacy/
│   ├── dp.py            Per-sample clipping, Gaussian noise, DP accountant
│   ├── he.py            TenSEAL CKKS session management
│   └── accountant.py    Exact integer-order RDP accountant
└── proto/               Protobuf definition + generator
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate the synthetic BraTS-style dataset and the hospital partition
python -m fedmed.prepare_data

# Generate the TLS certificate authority + server/client certificates
python -m fedmed.tls
```

### 2. Frontend

```bash
cd frontend
npm install
```

### 3. Run everything

**Option A — one command (recommended).** Starts the FastAPI API, the Flower aggregator, and all three hospital nodes:

```bash
cd backend
python -m fedmed.api        # REST + WebSocket on :8000
```

Then trigger a run from the dashboard (**Training** page), or from the CLI:

```bash
# Centralized baseline
python -m fedmed.baseline

# Full federated run (10 rounds)
python -m fedmed.orchestrator --rounds 10

# Federated run where hospital-3 drops offline at round 3
python -m fedmed.orchestrator --rounds 10 --dropout-round 3
```

**Option B — separate processes** (closer to a real cross-silo deployment):

```bash
# Terminal 1 — the aggregator
python -m fedmed.server --rounds 10

# Terminals 2-4 — the hospital nodes
python -m fedmed.node --hospital hospital-1
python -m fedmed.node --hospital hospital-2
python -m fedmed.node --hospital hospital-3
```

**Option C — standalone API + orchestrator** (the API and the aggregator need different ports):

```bash
python -m uvicorn fedmed.api:app --port 8000     # REST + WebSocket
python -m fedmed.orchestrator --rounds 10        # aggregator + nodes
```

### 4. Dashboard

```bash
cd frontend
npm run dev          # http://127.0.0.1:5173
```

| Port | Service |
| --- | --- |
| `5173` | React dashboard (Vite) |
| `8000` | FedMed REST API + WebSocket |
| `8001` | Flower aggregator (gRPC, TLS) |
| `9101-9103` | Hospital node gRPC/TLS services |

---

## 🔌 API Reference

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness probe |
| `GET` | `/api/status` | Run status, DP budget, progress, job registry |
| `GET` | `/api/nodes` | Per-hospital telemetry |
| `GET` | `/api/metrics` | Round history + baseline comparison |
| `POST` | `/api/train/baseline` | Launch the centralized baseline |
| `POST` | `/api/train/federated` | Launch the federated run |
| `GET` | `/api/mri/analyze` | Inference + rendered tumour masks |
| `GET` | `/api/privacy` | HE/DP posture + audit logs |
| `GET`/`POST` | `/api/settings` | Effective configuration / persist overrides |
| `WS` | `/ws/metrics` | Live metric stream |

---

## 🧪 Tests & Benchmarks

```bash
cd backend
python -m pytest tests/ -v
```

| Suite | Covers |
| --- | --- |
| `test_he_shared_context.py` | Shared-context CKKS aggregation (regression) |
| `test_dp.py` | Per-sample clipping, noise, accountant monotonicity |
| `test_accountant.py` | RDP bounds, $\varepsilon$ growth, $\delta$ handling |
| `test_core.py` | Model shapes, dataset partition, metrics |
| `test_federated.py` | End-to-end Flower round |

Benchmarks and calibration sweeps live alongside them:

```bash
python -m tests.bench_he      # CKKS payload size vs modulus chain
python -m tests.bench_dp      # DP overhead
python -m tests.sweep3        # (σ, lr) calibration sweep
```

---

## 📈 Dashboard Pages

| Page | Contents |
| --- | --- |
| **Dashboard** | Headline stats, dual-axis Dice/Loss chart, node grid, live MRI preview |
| **Hospitals** | Per-node cards: status, samples, encrypted flag, payload size, local Dice |
| **Training** | Launch baseline/federated runs, configure the dropout round, per-round table |
| **MRI Analysis** | Rendered MRI slice + tumour-mask overlay, per-class Dice, dataset distribution |
| **Privacy & Security** | HE posture, DP budget, live privacy & security audit log |
| **Settings** | Effective configuration, persisted overrides |

---

## 📊 Dataset

The real BraTS dataset requires credentialed access and hundreds of gigabytes, so FedMed generates a **structurally faithful surrogate**: 3D MRI-like volumes with a nested core/enhancing tumour, shaped like real BraTS pre-processing output (cropped/resampled to `64³` patches).

Each hospital receives a distinct **scanner bias** (multiplicative gain + additive offset), so the partition is genuinely non-IID — without distribution shift FedAvg would trivially match the centralized baseline and the whole audit would be meaningless.

| Hospital | Node ID | Train / Val | Split | Scanner bias |
| --- | --- | --- | --- | --- |
| hospital-1 | HOSP-001 | 48 / 15 | 40 % | −0.35 |
| hospital-2 | HOSP-002 | 42 / 12 | 35 % | +0.05 |
| hospital-3 | HOSP-003 | 30 / 9 | 25 % | +0.42 |

> ⚠️ No real medical data is stored in this repository.

---

## 🛠️ Technology Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.11+ / JavaScript (ESM) |
| Deep learning | PyTorch 2.10, MONAI 1.6 |
| Federated learning | Flower 1.42 |
| Homomorphic encryption | TenSEAL 0.3 (CKKS) |
| Differential privacy | Custom DP-SGD + RDP accountant |
| Communication | gRPC + TLS (mTLS on node control plane), Protobuf |
| API | FastAPI, Uvicorn, WebSocket |
| Frontend | React 19, Vite 8, Recharts 3 |

---

## ⚠️ Known Limitations

- **Key Management & Cryptographic Authority.** The aggregator generates the CKKS key pair and retains the secret key, meaning it has the technical capability to decrypt individual client updates. Nodes receive a public-only context. A zero-trust production deployment would require an asymmetric threshold homomorphic scheme (e.g., threshold CKKS) where the secret key is shared across hospitals.
- **The synthetic dataset** is a structural surrogate for BraTS, not clinical data. Dice scores here are not comparable to published BraTS benchmarks.
- **The privacy budget is large** (ε ≈ 297 over 10 rounds). Tightening it requires fewer rounds, a smaller sampling rate, or a larger `σ` — each of which trades accuracy for privacy.
- **Single-process simulation.** The hospital nodes share a host; a real deployment would isolate them.

---

### 📜 License

Developed for **educational and research purposes**.
