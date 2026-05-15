# D9 Task 4 — Wall-Clock Timing

**Platform:** gpu  
**N:** 48  
**dtype:** f32  
**Warm-up iters:** 3  
**Timed iters:** 20  
**Date:** 20260515  
**JAX:** 0.7.2

Each call is `f(u_tiles, v_tiles)` followed by `block_until_ready()` on both outputs. Timing via `time.perf_counter()`. Compile cost paid in warm-up and excluded.

| Statistic | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| median (ms) | 0.1168 | 0.6646 | 1.8191 |
| min (ms) | 0.1111 | 0.6523 | 1.7574 |
| max (ms) | 0.1646 | 0.7103 | 1.9238 |
| mean (ms) | 0.1228 | 0.6719 | 1.8235 |
| std (ms) | 0.0158 | 0.0177 | 0.0385 |

Per-iteration ms times are saved alongside this summary as `<variant>.times` (one float per line).
