# Galewsky N-sweep timing

**Platform:** tpu (`TPU_0(process=0,(0,0,0,0))`)  
**Date:** 2026-05-18T15:58:50.392569+00:00  
**JAX:** 0.7.2  
**jaxlib:** 0.7.2  
**Git:** db7222e6f29f010dc62bcd8d44f516bcf88e6404 (jax-port)  
**Host:** 8f68bfe7d0c1  
**Exchange:** `matrix_form` (`MatrixFormExchange`) — auto-dispatch resolved on this platform

Config: dt = 300.0 × (48 / N) for CFL ≈ 0.6 across the sweep; perturbed Galewsky IC; narrow ∇⁴ hyperviscosity (C=0.005); 3 warmup + 1000 timed steps; **batched throughput** (one block at end, mean = total/iters) — matches M5+ 6-day-run methodology.

| N | dt (s) | compile (s) | timed steps | total (s) | ms/step |
|---:|---:|---:|---:|---:|---:|
| 48 | 300.000 | 107.98 | 1000 | 3.152 | 3.1519 |
| 64 | 225.000 | 108.43 | 1000 | 3.469 | 3.4686 |
| 96 | 150.000 | 111.96 | 1000 | 4.705 | 4.7046 |
| 128 | 112.500 | 115.89 | 1000 | 7.017 | 7.0170 |
| 192 | 75.000 | 121.05 | 1000 | 9.320 | 9.3204 |
| 256 | 56.250 | 137.45 | 1000 | 13.443 | 13.4432 |
| 384 | 37.500 | 200.64 | 1000 | 20.620 | 20.6203 |

