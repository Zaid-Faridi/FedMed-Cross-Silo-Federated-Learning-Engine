"""Verify the shared context distribution flow (server -> node -> server)."""

from __future__ import annotations

import numpy as np
import tenseal as ts


def make_ctx():
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = float(2**30)
    return ctx


def main():
    n = 8192
    v = np.random.randn(n) * 0.01
    w = np.random.randn(n) * 0.01

    server_ctx = make_ctx()

    # Node receives the serialized context (over TLS) and encrypts with it.
    node_ctx = ts.Context.load(server_ctx.serialize())
    node_ctx.global_scale = float(2**30)
    print("node ctx loaded, scale:", node_ctx.global_scale)

    ct = ts.ckks_vector(node_ctx, v).serialize()
    print("node ciphertext size:", f"{len(ct):,}")

    # Server loads the ciphertext with its own context and decrypts.
    loaded = ts.CKKSVector.load(server_ctx, ct)
    err = float(np.abs(np.array(loaded.decrypt()) - v).max())
    print(f"node-encrypted -> server-decrypt max err: {err:.2e}")

    # Two nodes, pre-scaled plaintext, server sums ciphertexts.
    c1 = ts.ckks_vector(node_ctx, v * 0.4).serialize()
    c2 = ts.ckks_vector(node_ctx, w * 0.6).serialize()
    agg = ts.CKKSVector.load(server_ctx, c1) + ts.CKKSVector.load(server_ctx, c2)
    print(
        "server homomorphic sum max err:",
        float(np.abs(np.array(agg.decrypt()) - (v * 0.4 + w * 0.6)).max()),
    )

    # Check the asymmetric path once more but setting scale AFTER public-ify.
    print("\n=== asymmetric, scale set after make_context_public ===")
    srv = make_ctx()
    cli = ts.Context.load(srv.serialize())
    cli.make_context_public()
    cli.global_scale = float(2**30)
    ct2 = ts.ckks_vector(cli, v).serialize()
    back = ts.CKKSVector.load(srv, ct2)
    print("err:", float(np.abs(np.array(back.decrypt()) - v).max()))


if __name__ == "__main__":
    main()
