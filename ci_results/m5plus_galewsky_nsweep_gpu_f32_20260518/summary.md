# Galewsky N-sweep timing

**Platform:** gpu (`cuda:0`)  
**Date:** 2026-05-18T15:45:00.825905+00:00  
**JAX:** 0.7.2  
**jaxlib:** 0.7.2  
**Git:** db7222e6f29f010dc62bcd8d44f516bcf88e6404 (jax-port)  
**Host:** 8c971208ba6c  
**Exchange:** `index_scatter` (`IndexScatterExchange`) — auto-dispatch resolved on this platform

Config: dt = 300.0 × (48 / N) for CFL ≈ 0.6 across the sweep; perturbed Galewsky IC; narrow ∇⁴ hyperviscosity (C=0.005); 3 warmup + 1000 timed steps; **batched throughput** (one block at end, mean = total/iters) — matches M5+ 6-day-run methodology.

| N | dt (s) | compile (s) | timed steps | total (s) | ms/step |
|---:|---:|---:|---:|---:|---:|
| 48 | 300.000 | 35.83 | 1000 | 1.296 | 1.2960 |
| 64 | 225.000 | 35.39 | 1000 | 1.422 | 1.4215 |
| 96 | 150.000 | 36.56 | 1000 | 1.700 | 1.7000 |
| 128 | 112.500 | 35.66 | 1000 | 2.037 | 2.0374 |
| 192 | 75.000 | 36.48 | 1000 | 3.016 | 3.0159 |
| 256 | 56.250 | 47.49 | 1000 | 4.297 | 4.2974 |
| 384 | 37.500 | 44.76 | 1000 | 8.936 | 8.9359 |

