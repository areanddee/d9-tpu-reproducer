# Galewsky N-sweep timing

**Platform:** tpu (`TPU_0(process=0,(0,0,0,0))`)  
**Date:** 2026-05-19T15:35:32.167002+00:00  
**JAX:** 0.7.2  
**jaxlib:** 0.7.2  
**Git:** da2b231214e4e79da792800f5c5bb0c2edb92cbe (jax-port)  
**Host:** 6b6ff818b8b6  
**Exchange:** `dense_gather` (`DenseGatherExchange`) — explicit

Config: dt = 300.0 × (48 / N) for CFL ≈ 0.6 across the sweep; perturbed Galewsky IC; narrow ∇⁴ hyperviscosity (C=0.005); 3 warmup + 1000 timed steps; **batched throughput** (one block at end, mean = total/iters) — matches M5+ 6-day-run methodology.

| N | dt (s) | compile (s) | timed steps | total (s) | ms/step |
|---:|---:|---:|---:|---:|---:|
| 48 | 300.000 | 101.25 | 1000 | 9.599 | 9.5988 |
| 64 | 225.000 | 100.04 | 1000 | 11.674 | 11.6739 |
| 96 | 150.000 | 96.50 | 1000 | 17.240 | 17.2396 |
| 128 | 112.500 | 97.33 | 1000 | 23.567 | 23.5671 |
| 192 | 75.000 | 101.49 | 1000 | 36.691 | 36.6908 |
| 256 | 56.250 | 109.37 | 1000 | 48.000 | 48.0005 |
| 384 | 37.500 | 140.25 | 1000 | 71.732 | 71.7319 |

