# D9 Task 4 — Wall-Clock Timing

**Platform:** cpu  
**N:** 192  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260514  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 6.7415 | 1.7796 | 330.7162 |
| min (ms) | 6.3310 | 1.6369 | 320.9789 |
| max (ms) | 10.5930 | 1.9611 | 358.3445 |
| mean (ms) | 6.9204 | 1.7805 | 334.5696 |
| std (ms) | 0.8618 | 0.0911 | 9.9604 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
