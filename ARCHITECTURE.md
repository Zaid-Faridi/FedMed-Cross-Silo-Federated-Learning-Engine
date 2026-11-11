# FedMed Architecture & System Design

This document describes the software architecture, network interfaces, cryptographic data flows, and parameter configurations of the FedMed cross-silo federated learning platform. All parameter values documented below are drawn directly from [`backend/fedmed/config.py`](file:///Users/zaidfaridi/Downloads/PROJECTS%20/FedMed/backend/fedmed/config.py).

---

## 1. System Overview & Component Hierarchy

FedMed simulates a multi-institution cross-silo federated learning deployment where three hospital nodes collaboratively train a 3D U-Net segmentation model without exchanging raw patient data.

```text
┌──────────────────────────────────────────────────────────────────┐
│                   React 19 + Vite UI (:5173)                     │
│          REST client + WebSocket stream (/ws/metrics)            │
└─────────────────────────────────┬────────────────────────────────┘
                                  │ HTTP / WS
                                  ▼
┌──────────────────────────────────────────────────────────────────┐
│              Aggregator Host Runtime & Transport                 │
│                                                                  │
│  FastAPI Application (:8000)                                     │
│  ├── REST API routes (/api/status, /api/nodes, /api/metrics...)  │
│  ├── /ws/metrics WebSocket server                                │
│  └── RuntimeState & EventBus (pub-sub bridge)                    │
│                                                                  │
│  Flower Server (:8001, mutual TLS)                               │
│  ├── FederatedStrategy (FedAvg over plaintext-scaled deltas)     │
│  ├── CKKS Server Context (holds secret + evaluation keys)        │
│  └── RDP Accountant (integer-order Rényi privacy accounting)     │
└───────────────────▲─────────────▲─────────────▲──────────────────┘
                    │             │             │
                    │ gRPC (mTLS) │ gRPC (mTLS) │ gRPC (mTLS)
                    │ Flower :8001│ Flower :8001│ Flower :8001
                    ▼             ▼             ▼
       ┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
       │   Hospital Node 1      │ │   Hospital Node 2      │ │   Hospital Node 3      │
       │   node_id: HOSP-001    │ │   node_id: HOSP-002    │ │   node_id: HOSP-003    │
       │   Flower Client        │ │   Flower Client        │ │   Flower Client        │
       │   gRPC Control: :9101  │ │   gRPC Control: :9102  │ │   gRPC Control: :9103  │
       │   Local Dataset Slice  │ │   Local Dataset Slice  │ │   Local Dataset Slice  │
       │   DP-SGD Optimizer     │ │   DP-SGD Optimizer     │ │   DP-SGD Optimizer     │
       │   CKKS Encryption      │ │   CKKS Encryption      │ │   CKKS Encryption      │
       └────────────────────────┘ └────────────────────────┘ └────────────────────────┘
```

### Components
1. **React 19 Frontend (`frontend/`):** Single-page web dashboard using Vite 8, Recharts 3, and native CSS. Provides real-time training curves, per-hospital telemetry cards, axial MRI slice/tumor mask viewers, and privacy budget indicators.
2. **FastAPI Coordination Layer (`backend/fedmed/api.py`):** Runs on port `8000`. Exposes REST management endpoints and a WebSocket channel (`/ws/metrics`) that receives events published from training threads via a thread-safe `EventBus`.
3. **Flower Aggregator (`backend/fedmed/server.py`, `backend/fedmed/strategy.py`):** Runs on port `8001` over mutual TLS. Coordinates federated training rounds using a custom `FederatedStrategy`. It generates the CKKS cryptographic context, distributes the public key material, receives client updates, sums encrypted deltas, decrypts the aggregate, and updates global model parameters.
4. **Hospital Nodes (`backend/fedmed/client.py`, `backend/fedmed/node.py`):** Three simulated silos (`HOSP-001`, `HOSP-002`, `HOSP-003`). Each node manages:
   * A private partition of synthetic 3D MRI volumes.
   * A local DP-SGD training loop with per-sample gradient clipping and calibrated Gaussian perturbation.
   * Plaintext weight scaling followed by CKKS encryption.
   * A standalone gRPC control service (`backend/fedmed/grpc_service.py`) on dedicated node ports.

---

## 2. Network Topology & Port Bindings

As defined in `ServerConfig` and `HospitalConfig` in `backend/fedmed/config.py`:

| Service | Host / Interface | Port | Transport Protocol | Authentication & Security |
| :--- | :--- | :--- | :--- | :--- |
| **Frontend UI** | `localhost` | `5173` | HTTP | None (local development server) |
| **FedMed REST / WebSocket API** | `127.0.0.1` | `8000` | HTTP / WebSocket | CORS enabled |
| **Flower Aggregator** | `127.0.0.1` | `8001` | gRPC | Mutual TLS (mTLS) with Local CA validation |
| **Hospital Node 1 Control Plane** | `127.0.0.1` | `9101` | gRPC | Mutual TLS (mTLS) with Local CA validation |
| **Hospital Node 2 Control Plane** | `127.0.0.1` | `9102` | gRPC | Mutual TLS (mTLS) with Local CA validation |
| **Hospital Node 3 Control Plane** | `127.0.0.1` | `9103` | gRPC | Mutual TLS (mTLS) with Local CA validation |
| **Default Node Coordination** | `127.0.0.1` | `50051` | gRPC | Mutual TLS default fallback port |

---

## 3. Machine Learning & Volumetric Segmentation Architecture

### Model Architecture (`ModelConfig`)
* **Base Architecture:** MONAI 3D `UNet` (`spatial_dims=3`).
* **Input Channels:** `1` (single-channel volumetric MRI patch).
* **Output Channels:** `3` segmentation classes:
  * Class 0: Background
  * Class 1: Edema
  * Class 2: Enhancing tumor
* **Patch Size:** `(64, 64, 64)`.
* **Channel Multipliers:** `(16, 32, 64, 128)`.
* **Strides:** `(2, 2, 2)`.
* **Residual Units:** `1` per level.
* **Normalization & Regularization:** Instance normalization (`norm="INSTANCE"`), dropout rate `0.2`.
* **Trainable Parameters:** Exactly `600,002` parameters.

### Optimization & Loss
* **Loss Function:** `dice_ce_loss` combining soft Dice loss over foreground channels (classes 1 and 2) and Cross-Entropy over all classes (background included).
* **Optimizer:** Plain Stochastic Gradient Descent (SGD) with momentum:
  * `learning_rate = 0.05`
  * `momentum = 0.9`
  * `weight_decay = 1e-5`
* **Local Epochs per Round:** `2`.
* **Batch Size:** `8`.
* **Total Federated Rounds:** `10`.

---

## 4. Privacy & Cryptographic Configuration

Parameters are configured via `PrivacyConfig` in `backend/fedmed/config.py`:

### Homomorphic Encryption (TenSEAL CKKS)
* **Scheme:** CKKS (Cheon-Kim-Kim-Song).
* **Polynomial Modulus Degree (`he_poly_modulus`):** `16384`.
* **Coefficient Modulus Bit Sizes (`he_coeff_modulus`):** `(60, 40, 40, 60)`.
* **Global Scale:** $2^{30}$ (approx. 30 bits of fractional precision).
* **Payload Footprint & Expansion:**
  * For 600,002 model parameters, float32 plaintext occupies $\approx 2.40\text{ MB}$ ($2,400,008\text{ bytes}$) and float64 occupies $\approx 4.80\text{ MB}$ ($4,800,016\text{ bytes}$).
  * A serialized CKKS ciphertext vector under the above parameters produces a payload of $\approx 48.7\text{ MB}$ ($\approx 48,700,000\text{ bytes}$).
  * This represents an expansion factor of $\approx \mathbf{20.3\times}$ relative to raw float32 parameter buffers ($\approx 10.1\times$ relative to float64 buffers).
* **Aggregation Arithmetic:**
  * To avoid losing precision from homomorphic scalar multiplication, nodes pre-scale their local weight update delta $\Delta_i$ by their FedAvg weight $w_i = \frac{n_i}{\sum n}$ in plaintext prior to encryption.
  * The aggregator receives ciphertexts $\text{Enc}(w_i \Delta_i)$ and evaluates ciphertext addition:
    $$\mathbf{c}_{\text{global}} = \bigoplus_{i=1}^K \text{Enc}(w_i \Delta_i)$$
  * Ciphertext addition is exact in CKKS, maintaining precision with $\le 4 \times 10^{-6}$ absolute error.

### Differential Privacy (DP-SGD & RDP Accountant)
* **Mechanism:** Sampled Gaussian mechanism with per-sample gradient evaluation.
* **Per-Sample L2 Clip Norm (`dp_clip_norm`):** `1.0`.
* **Noise Multiplier (`dp_noise_multiplier`):** `0.5` ($\sigma = 0.5 \times C = 0.5$).
* **Target Delta (`dp_delta`):** `1e-5`.
* **Nominal Subsampling Ratio (`dp_sample_rate`):** `0.16` ($q \approx \frac{\text{batch\_size}}{\text{local\_dataset\_size}}$).
* **Accountant:** Rényi Differential Privacy (RDP) based on integer-order bounds evaluated in log-space across orders $\alpha \in [2, 64] \cup \{128, 256, 512\}$.

---

## 5. End-to-End Data Flow

```text
1. Round Initialization:
   Flower Aggregator creates FitIns containing current global parameters W_t
   and serialized CKKS public context blob.

2. Client Distribution:
   FitIns is transmitted to Hospital Nodes 1, 2, and 3 via mTLS gRPC connections.

3. Local Computation on Nodes:
   a. Node loads W_t into local 3D U-Net.
   b. For each local minibatch, gradients are calculated per sample.
   c. Each individual sample gradient is clipped to L2 norm <= 1.0.
   d. Clipped gradients are summed, and Gaussian noise N(0, (0.5)^2 * I) is added.
   e. Parameter delta is computed: Delta_i = W_local - W_t.
   f. Delta is pre-scaled by client sample fraction: w_i * Delta_i.
   g. Scaled delta is encrypted into CKKS ciphertext blob c_i (~48.7 MB).

4. Upload:
   FitRes containing ciphertext c_i and metadata is returned to Aggregator.

5. Aggregation & Model Update:
   a. Aggregator performs homomorphic addition: c_sum = c_1 + c_2 + c_3.
   b. Aggregator decrypts c_sum using server private key to obtain Delta_global.
   c. If a node dropped offline, Aggregator renormalizes Delta_global by the
      sum of received weights.
   d. Aggregator applies Delta_global to update W_(t+1) = W_t + Delta_global.
   e. Privacy accountant logs step and increments cumulative epsilon.
   f. Round metrics are emitted to EventBus and pushed to frontend via WebSocket.
```
