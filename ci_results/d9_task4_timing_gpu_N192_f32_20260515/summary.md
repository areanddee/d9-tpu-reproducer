# D9 Task 4 — Wall-Clock Timing

**Platform:** gpu  
**N:** 192  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260515  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 0.1472 | 0.7077 | 1.6690 |
| min (ms) | 0.1381 | 0.6855 | 1.6351 |
| max (ms) | 0.1746 | 0.7375 | 1.7309 |
| mean (ms) | 0.1520 | 0.7109 | 1.6751 |
| std (ms) | 0.0113 | 0.0132 | 0.0246 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
