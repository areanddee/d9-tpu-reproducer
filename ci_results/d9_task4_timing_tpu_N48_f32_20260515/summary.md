# D9 Task 4 — Wall-Clock Timing

**Platform:** tpu  
**N:** 48  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260515  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 2.7361 | 1.0854 | 16.7040 |
| min (ms) | 2.7182 | 1.0737 | 16.6759 |
| max (ms) | 2.7670 | 1.1228 | 16.7679 |
| mean (ms) | 2.7385 | 1.0870 | 16.7149 |
| std (ms) | 0.0136 | 0.0121 | 0.0305 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
