# D9 Task 3 — HLO Op Counts

**Platform:** gpu  
**N:** 48  
**dtype:** f32  
**Date:** 20260515  
**JAX:** 0.7.2

Pre-optimization StableHLO op counts via `jax.jit(f).lower(*args).as_text()`. Each cell is a substring count of `stablehlo.<op>` in the lowered IR.

| Op | IndexScatter | DenseGather | vmap-of-DS |
| --- | ---: | ---: | ---: |
| `stablehlo.gather` | 8 | 192 | 1152 |
| `stablehlo.scatter` | 4 | 0 | 288 |
| `stablehlo.dynamic_slice` | 0 | 96 | 0 |
| `stablehlo.dynamic_update_slice` | 0 | 144 | 0 |
| `stablehlo.dot_general` | 0 | 0 | 0 |
| `stablehlo.reduce` | 4 | 96 | 288 |
| `stablehlo.concatenate` | 6 | 96 | 876 |
| **total HLO lines** | **223** | **5718** | **16001** |

Full StableHLO per variant is saved alongside this summary as `<variant>.stablehlo`.
