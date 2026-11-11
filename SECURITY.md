# FedMed Threat Model & Security Specification

This document provides a rigorous, transparent threat model for FedMed. It defines the trust boundaries, cryptographic defenses, structural assumptions, and explicit security non-guarantees of the system.

---

## 1. Security Architecture & Threat Model Overview

FedMed coordinates federated training of volumetric MRI segmentation models across three simulated hospital silos. The security model balances institutional privacy with collaborative deep learning.

### Trust Boundary Diagram
```text
┌────────────────────────────────────────────────────────────┐
│                    Aggregator Boundary                     │
│  - Generates CKKS key pair (holds secret key SK and pk)   │
│  - Mints CA, server, and client certificates               │
│  - Aggregates updates and decrypts final sum               │
└─────────────────────────────▲──────────────────────────────┘
                              │
               Flower Port 8001: Server-side TLS
             Node Control Ports (9101-9103): mTLS
              Shared client cert on control plane
                              │
┌─────────────────────────────▼──────────────────────────────┐
│                  Hospital Node Boundary                    │
│  - Receives public-only CKKS context (pk)                  │
│  - Holds private MRI dataset slice                         │
│  - Applies local DP-SGD (clipping + Gaussian noise)        │
│  - Encrypts update under public key                        │
└────────────────────────────────────────────────────────────┘
```

---

## 2. What Is Trusted

1. **Central Aggregator Host:**
   * The central aggregator is trusted to execute the documented protocol correctly.
   * **Key Authority:** The central aggregator mints the CKKS key pair and **retains the secret key** ($SK$).
   * **Protocol Compliance:** The aggregator is trusted not to maliciously decrypt individual client ciphertexts before addition, although it possesses the cryptographic capability to do so.
2. **Local Certificate Authority:**
   * The CA keys (`ca.key`, `ca.crt`) generated during setup are assumed uncompromised.
3. **Local Client Environments:**
   * Each hospital node is assumed to execute authentic training code on genuine patient data.
   * Node processes are trusted to accurately compute sample counts ($n_i$) used for client-side weighting.

---

## 3. What Each Layer Defends Against

### A. Federated Partitioning
* **Defends Against:** Direct exposure of raw patient scans, medical metadata, and protected health information (PHI). Raw MRI voxels never leave the local hospital environment; only model weight deltas are transmitted.

### B. Differential Privacy (DP-SGD)
* **Defends Against:** Membership inference attacks, model inversion, and training data extraction from model gradients.
* **Mechanism:** Per-patient gradient clipping ($C = 1.0$) ensures no individual scan dominates an update. Calibrated Gaussian noise ($\sigma = 0.5$) mathematically bounds the privacy leakage across rounds according to Rényi Differential Privacy ($(\epsilon \approx 297, \delta = 10^{-5})$ over 10 rounds).

### C. Homomorphic Encryption (CKKS)
* **Defends Against:** Network eavesdroppers and intermediate transport proxies inspecting plain weight deltas.
* **Mechanism:** Model updates are encrypted under a shared public evaluation context before leaving the node. Ciphertexts are summed in ciphertext space.

### D. Transport Security (Server-side TLS & Node Control mTLS)
* **Defends Against:** Man-in-the-middle (MITM) network tampering, packet inspection, and unauthorized plaintext traffic injection on gRPC channels.
* **Flower Training Channel (port 8001):** Uses server-side TLS (RSA-2048). Flower's server verifies server identity to connecting clients using the local CA, encrypting traffic in transit; however, Flower's internal gRPC server sets `require_client_auth=False`, meaning client certificates are not verified on port 8001.
* **Node Control Plane (ports 9101-9103):** FedMed's node control services enforce mutual TLS (mTLS) with `require_client_auth=True`.

---

## 4. Explicit "Not Protected" List (Vulnerabilities & Limitations)

The following attack vectors and threat scenarios are **explicitly not protected** by the current FedMed implementation:

1. **Aggregator Insider Threat / Compromised Aggregator:**
   * **The aggregator holds the secret key ($SK$) of the CKKS scheme.** Because it possesses the decryption key, the aggregator **can decrypt any single hospital update** if it chooses to do so.
   * Hospital nodes receive a public-only context (`make_context_public()`), so nodes cannot decrypt each other's updates. However, privacy against the aggregator relies strictly on protocol adherence (the aggregator chooses to sum before decrypting).
   * In a zero-trust production deployment, an asymmetric threshold homomorphic encryption scheme (such as threshold CKKS in OpenFHE or Lattigo) is required, where the secret key is split among the hospital silos and never held by the central server.

2. **Malicious Clients & Byzantine Model Poisoning:**
   * **There is no defense against malicious clients.** Because updates arrive as opaque CKKS ciphertexts, the aggregator **cannot inspect** gradient distributions, detect backdoors, or identify data poisoning attacks.
   * Byzantine-robust aggregation algorithms (such as Krum, Trimmed Mean, or coordinate-wise median) cannot be evaluated over CKKS ciphertexts without expensive non-linear cryptographic circuits.

3. **Flower Training Channel Does Not Require Client Authentication (Server-Side TLS Only):**
   * The Flower federated learning training channel (`127.0.0.1:8001`) operates with **server-side TLS**, not mutual TLS. Although the aggregator is provided root certificates, Flower's internal `flwr.supercore.grpc.generic_create_grpc_server` sets `require_client_auth=False`.
   * As verified by empirical connection probing, clients can connect to port 8001 presenting only the CA certificate with no client certificate.
   * Only the separate hospital node control services (ports `9101`-`9103`) strictly enforce mTLS (`require_client_auth=True`).
   * On port 9101-9103 mTLS, all hospital nodes share a **single client certificate** (`client.crt`, `client.key`) issued with Common Name `fedmed-node`, so transport-layer caller identity is cluster-wide rather than per-hospital.

4. **Self-Reported Node Identity:**
   * Hospital node identity (`hospital_id`, `node_id`, `samples`) is self-reported over Flower's `get_properties` hook and gRPC metadata.
   * A compromised node could spoof another hospital's identity or misreport sample counts to distort FedAvg weights.

5. **Post-Quantum Security Status:**
   * The lattice-based CKKS homomorphic encryption scheme is based on the Ring Learning with Errors (RLWE) problem and is arguably post-quantum secure.
   * **The network transport layer is not post-quantum secure:** mTLS connections rely on standard RSA-2048 and classical Diffie-Hellman exchanges, which are vulnerable to cryptographically relevant quantum computers.

6. **Dropout Renormalization Visibility:**
   * When a hospital drops offline mid-round, the server renormalizes the decrypted aggregate by the sum of received weights. While this avoids multiplying ciphertexts by scalars, the aggregator possesses the secret key throughout the process.
