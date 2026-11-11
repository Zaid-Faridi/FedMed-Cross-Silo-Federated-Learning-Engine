"""Benchmark forward/backward speed for candidate model configurations."""

from __future__ import annotations

import time

import torch
from monai.networks.nets import UNet


def bench(ch, st, res_units, size, batch, label=""):
    m = UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=ch, strides=st,
             num_res_units=res_units, dropout=0.2, norm="INSTANCE", bias=True, up_kernel_size=3)
    n = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(batch, 1, *size)
    y = torch.randint(0, 3, (batch, *size))

    # warmup
    for _ in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()

    t0 = time.time()
    iters = 5
    for _ in range(iters):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x), y)
        loss.backward()
        opt.step()
    dt = (time.time() - t0) / iters

    # per-sample grad cost (vmap-free loop)
    t0 = time.time()
    for i in range(2):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(m(x[i:i+1]), y[i:i+1])
        loss.backward()
    single = (time.time() - t0) / 2

    print(f"{label:28s} params={n:>9,}  batch_step={dt:5.3f}s  "
          f"per_sample={single:5.3f}s  he_size@20.3x={n*4*20.3/1e6:6.1f}MB")


if __name__ == "__main__":
    S = (64, 64, 64)
    bench((16, 32, 64, 128, 256), (2, 2, 2, 2), 2, S, 2, "A base(ch5,st4,ru2)")
    bench((16, 32, 64, 128), (2, 2, 2), 2, S, 2, "B ch4,st3,ru2")
    bench((16, 32, 64, 128), (2, 2, 2), 1, S, 2, "C ch4,st3,ru1")
    bench((8, 16, 32, 64), (2, 2, 2), 2, S, 2, "D ch4small,st3,ru2")
    bench((8, 16, 32, 64), (2, 2, 2), 1, S, 2, "E ch4small,st3,ru1")
    bench((16, 32, 64), (2, 2, 2), 1, S, 2, "F ch3,st3,ru1")
    print()
    bench((16, 32, 64, 128), (2, 2, 2), 2, (48, 48, 48), 2, "G ch4,st3,ru2 @48")
    bench((8, 16, 32, 64), (2, 2, 2), 1, (48, 48, 48), 2, "H ch4small,st3,ru1 @48")
