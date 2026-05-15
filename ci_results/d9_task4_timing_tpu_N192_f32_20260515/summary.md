# D9 Task 4 — Wall-Clock Timing

**Platform:** tpu  
**N:** 192  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260515  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 24.2407 | 3.6858 | 80.4070 |
| min (ms) | 24.2093 | 3.6728 | 80.2043 |
| max (ms) | 24.2835 | 3.7100 | 80.5102 |
| mean (ms) | 24.2433 | 3.6891 | 80.3506 |
| std (ms) | 0.0190 | 0.0108 | 0.1108 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
