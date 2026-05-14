# D9 Task 4 — Wall-Clock Timing

**Platform:** cpu  
**N:** 48  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260514  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 1.0347 | 0.5008 | 3.9344 |
| min (ms) | 0.9119 | 0.4448 | 3.3653 |
| max (ms) | 1.4699 | 0.5515 | 15.2902 |
| mean (ms) | 1.0601 | 0.5016 | 4.7693 |
| std (ms) | 0.1260 | 0.0324 | 2.7117 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
