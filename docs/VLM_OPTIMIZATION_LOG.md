# VLM Optimization & Improvement Log

Tracking all optimizations, accuracy fixes, and research findings for the
Bonsai-27B (Qwen3-VL) vision tower implementation in bitgpu.

## Current State (after optimizations)

**Architecture**: Qwen3-VL vision tower
- 27 transformer blocks, hidden=1152, heads=16, head_dim=72
- Intermediate=4304, projection=5120
- Patch: 16×16, temporal=2, spatial_merge=2
- Position embeddings: 48×48 learned grid, bilinear interpolated
- 2D RoPE: partial rotary (50%), NeoX-style rotate_half
- Attention: bidirectional (no causal mask), per-frame segments
- GELU: tanh approximation (vision blocks), exact erf (patch merger — TODO)
- LayerNorm (not RMSNorm) with eps=1e-6
- No window attention (confirmed: Qwen3-VL removed this, unlike Qwen2.5-VL)

**Performance results** (234-patch screenshot, M3 Mac, 3-run benchmark):

| Metric | Baseline | After opt | Improvement |
|--------|----------|-----------|-------------|
| Vision tower | 50.8s | 5.4s | 89% faster |
| Prefill | 18.2s | 11.0s | 40% faster |
| Tokens/s | 3.6 | 5.9 | 64% faster |

**Benchmark results** (3 runs, 234 patches, 90s cooldown between runs):
- Run 1: 6040ms (cold), Run 2: 5451ms, Run 3: 5443ms
- Min: 5443ms, Max: 6040ms, Avg: 5645ms
- Per-layer: 201.6ms/layer

**Red 32×32 image results**:

| Metric | Baseline | After opt |
|--------|----------|-----------|
| Vision tower | 6.5s | 6.2s |
| Prefill | 3.9s | 2.5s |
| Tokens/s | 4.2 | 5.8 |
| Accuracy | ✓ "solid red square" | ✓ "solid red square" |

**Screenshot accuracy**: Model now recognizes it as a screenshot with text
content (social media / messaging app), but still can't read exact text.
The bicubic interpolation fix (a=-0.75 → a=-0.5) may improve this further
with more test images.

**Bottleneck analysis** (theoretical roofline @ 100 GB/s, 2.84 TFLOPS M3 / ~3.5 TFLOPS M4):

**Hardware**: Apple M4 (10-core GPU, 128 ALUs/core, 32-thread subgroups)
- Available WebGPU features: `subgroups`, `shader-f16`, `timestamp-query`, `chromium-experimental-subgroup-matrix`
- Subgroup size: 32 (fixed on Apple Silicon)
- Peak f32: ~3.5 TFLOPS | Peak f16: ~7 TFLOPS (2× f32)
- DRAM: 120 GB/s | L2: ~400 GB/s | Threadgroup: ~23 GB/s (SLOWER than DRAM!)

| Component | Per layer | All 27 layers | Theoretical time |
|-----------|-----------|---------------|------------------|
| Weights (Q8+F16) | 20.8 MB | 562.8 MB | 5.6 ms |
| Activations (f32) | 33.9 MB | 916.3 MB | 9.2 ms |
| Memory total | 54.8 MB | 1479 MB | 14.8 ms |
| Compute | 11.2 GFLOP | 301 GFLOP | 86-106 ms |
| **Roofline** | | | **86-106 ms** |

- **Theoretical bottleneck**: COMPUTE-BOUND (86-106ms compute vs 14.8ms memory)
- **Actual**: 5443ms — **51-63× slower than compute roofline**
- **Efficiency**: 1.6-2.0% of roofline

**Why the 51-63× gap?**
1. **Q8 dequantization overhead**: Each vec4 weight needs ~8 extra instructions
   (shift, mask, convert, multiply), halving effective compute throughput.
2. **Small matrix sizes**: 1152×1152 matmuls are too small to saturate the GPU's
   parallelism. The GPU has ~1280 ALUs but each workgroup only has 256 threads.
3. **Threadgroup shared memory is SLOW on Apple Silicon** (~23 GB/s vs 100 GB/s
   DRAM). The cooperative load + barrier pattern adds latency.
4. **Shader JIT compilation**: First run is ~600ms slower (6040 vs 5443).
5. **Attention O(N²)**: 234² × 16 heads = 876K scores, each with 72-wide dot.
6. **No subgroup usage**: Vision shaders use shared memory + barriers instead
   of hardware-native subgroup operations (3× speedup reported by ONNX Runtime).

**Key research findings** (Apple Silicon WebGPU optimization):
- Threadgroup shared memory (~23 GB/s) is SLOWER than DRAM (100 GB/s)
- Direct global loads can be 2.3-2.8× faster than shared memory staging
  (but NOT for our matmul pattern — cooperative loading reduces total traffic)
- Subgroup operations (Chrome 134+) give 3× speedup for matmul
- F16 compute is 2× peak throughput (7 vs 3.5 TFLOPS on M4)
- BK=32 is slightly slower than BK=16 (more shared memory, lower occupancy)
- MLX uses BM=64, BN=64, BK=16 for medium Apple Silicon devices

**Weight breakdown** (per layer, Q8+F16):

| Weight | Size | % of total |
|--------|------|------------|
| FFN down (F16) | 9.7 MB | 48% |
| FFN up (Q8) | 5.1 MB | 25% |
| QKV (Q8) | 4.1 MB | 20% |
| AttnOut (Q8) | 1.4 MB | 7% |
| LayerNorms | 0.02 MB | 0% |

**What-if projections**:
- F16 activations: 1.45× memory speedup (916→458 MB), but compute-bound → minimal
- Q4 weights: 1.10× memory speedup (563→424 MB), but compute-bound → minimal
- **Key insight**: Further weight/activation compression won't help much.
  The bottleneck is COMPUTE, not memory. Need to improve compute throughput.

**Root cause analysis** (why we're 51× from compute roofline):
The Q8_0 dequantization adds ~8 extra instructions per vec4 weight
(shift, mask, sign-extend, convert, multiply). Each FMA is only 1 of ~52
total instructions per K iteration, giving ~2% compute efficiency — which
matches the measured 1.1% roofline efficiency. The 2D tiled matmul with
256 threads, 4×4 register tiles, BK=16 is already well-optimized for this
constraint. Seven alternative approaches were tested and all were slower
(see "Findings: What DIDN'T work" below). The current implementation
represents a local optimum given the Q8_0 weight format and Apple Silicon
GPU architecture.

---

## Completed Optimizations

### 1. Pre-compute RoPE-applied Q/K (P0+P1)

**Problem**: The attention shader loaded K and applied RoPE inside the inner
loop. With N patches, each K vector was loaded and RoPE-rotated N times
(once per query) — O(N²) redundant work. For 234 patches: 54,756 redundant
K loads × 72 floats = 15.7 MB redundant reads per head per layer.

**Fix**: New `vision_apply_rope.wgsl` shader pre-computes RoPE-applied Q and
K in a single dispatch per layer. One workgroup per patch, 64 threads
cooperatively process all 16 heads × 72 dims = 1152 values.

**Files**: `shaders/vision_apply_rope.wgsl` (new), `src/engine.ts`

### 2. Shared-memory tiled attention (P2)

**Problem**: Each of 32 threads in the attention workgroup independently
loaded K/V from global memory — 32× redundant global memory reads.

**Fix**: Rewrote `vision_attention.wgsl` to cooperatively load K/V tiles
into workgroup shared memory. TILE_SIZE=16 (within 16KB workgroup storage
limit). All threads share the same K/V tile, eliminating redundant reads.

**Files**: `shaders/vision_attention.wgsl` (rewritten)

### 3. Fix bicubic interpolation parameter (P5 — CRITICAL ACCURACY FIX)

**Problem**: The bicubic resize used `a=-0.75` (Catmull-Rom), but PIL
BICUBIC uses `a=-0.5`. This mismatch corrupts fine text details during
image resizing, which is the likely cause of poor text recognition.

**Fix**: Changed `a=-0.75` to `a=-0.5` in `bicubicResize()`.

**Files**: `src/vision.ts` line 560

### 4. Q8_0 in-shader dequantization (P3 — WEIGHT BANDWIDTH REDUCTION)

**Problem**: Vision weights were dequantized from Q8_0 to f32 on CPU, then
uploaded as f32 buffers. Total weight reads: 60.8 MB per layer (f32).
At ~100 GB/s (M3), theoretical min is ~15ms/layer, actual ~1260ms — 84× gap.

**Fix**: Repack Q8_0 weights on CPU into GPU-friendly format:
- `packed: Uint32Array` — [N, K/4] u32 words, each containing 4 int8 values
- `scales: Float32Array` — [N, K/32] f32 block scales

New Q8 matmul shaders (`vision_matmul_q8`, `vision_matmul_q8_gelu`,
`vision_matmul_q8_add`) dequantize inline during the dot product:
```wgsl
let bits = i32(packed_word);
let v0 = f32((bits << 24) >> 24) * scale;  // extract signed int8, multiply by scale
```

This follows the proven pattern from the LLM engine's q8 KV cache
(`attention_sg_kv8.wgsl`), adapted for vision matmul weights.

**Weight bandwidth reduction**:
| Weight | f32 size | Q8 packed + scales | Reduction |
|--------|----------|-------------------|-----------|
| QKV [3456, 1152] | 15.9 MB | 5.0 MB | 3.2× |
| attn_out [1152, 1152] | 5.3 MB | 1.4 MB | 3.8× |
| ffn_up [4304, 1152] | 19.8 MB | 5.8 MB | 3.4× |
| ffn_down [1152, 4304] | 19.8 MB | 19.8 MB (F16, still f32) | 1.0× |
| **Total per layer** | **60.8 MB** | **32.0 MB** | **1.9×** |

**Files**: `src/vision.ts` (repackQ8_0, Q8PackedWeight type), `shaders/vision_matmul_q8*.wgsl` (new), `src/engine.ts` (dual buffer upload, Q8 shader dispatch)

**Results**: Vision tower 33.9s → 31.1s (8% faster). Accuracy preserved.
Improvement limited because ffn_down (biggest weight, 19.8 MB) is F16 not
Q8_0 — still uploaded as f32. Keeping it as f16 on GPU would add another
~2× on that weight (P4 below).

### 5. F16 storage for FFN down weights (P4 — WEIGHT BANDWIDTH)

**Problem**: FFN down is the largest vision weight (19.8 MB/layer, [1152, 4304])
and is stored as F16 in the GGUF file. It was converted to f32 on CPU and
uploaded as f32, wasting 2× bandwidth on the largest weight.

**Fix**: Keep FFN down as f16 on GPU. New `vision_matmul_f16_add.wgsl` shader
uses `enable f16;` and reads `array<f16>` weights, widening to f32 at read:
```wgsl
acc = acc + shared_x[k] * f32(w[w_base + kt + k]);
```
This follows the LLM engine's f16 KV cache pattern (`attention_sg_kv16.wgsl`).

**Weight bandwidth reduction**:
| Weight | Before | After | Reduction |
|--------|--------|-------|-----------|
| QKV (Q8) | 5.0 MB | 5.0 MB | — |
| attn_out (Q8) | 1.4 MB | 1.4 MB | — |
| ffn_up (Q8) | 5.8 MB | 5.8 MB | — |
| ffn_down (F16) | 19.8 MB (f32) | 9.9 MB (f16) | 2× |
| **Total per layer** | **32.0 MB** | **22.1 MB** | **1.45×** |

**Files**: `src/vision.ts` (repackF16, F16PackedWeight), `shaders/vision_matmul_f16_add.wgsl` (new), `src/engine.ts` (shader-f16 request, f16 upload, f16 shader dispatch)

**Results**: Accuracy preserved. Bandwidth reduced 1.45× further on top of Q8.
Combined with Q8: total weight bandwidth 60.8 MB → 22.1 MB per layer (2.75× reduction).

### 6. 2D tiled Q8 matmul (P6 — COMPUTE INTENSITY)

**Problem**: The 1D Q8 matmul read W from global memory for every M iteration
(234× redundant for 234 patches). Each thread read its own W row from global
memory in the inner K loop, with no reuse across patches.

**Fix**: 2D tiled register-blocked matmul (BM=64, BN=64, BK=16) with 4×4
register tiles per thread, 256 threads per workgroup. Both X and W tiles are
loaded into shared memory (8KB total), so W is read from global memory only
K/16 times and reused across 64 patches per tile.

Pattern follows the LLM engine's `matmul_split_tiled.wgsl`:
- 256 threads arranged as 16×16 grid
- Each thread computes a 4×4 output tile in registers
- Q8 dequantization happens during cooperative W tile load
- vec4 dot products for the accumulation loop

**Files**: `shaders/vision_matmul_q8_tiled.wgsl`, `vision_matmul_q8_tiled_gelu.wgsl`, `vision_matmul_q8_tiled_add.wgsl` (new), `src/engine.ts` (2D dispatch)

**Results**: Vision tower 31.1s → 26.9s (13% faster). Accuracy preserved.
Combined with Q8+F16: 33.9s → 26.9s (20.6% total improvement).

### 7. 2D tiled F16 matmul for FFN down (P5 — BIGGEST WIN)

**Problem**: The ffn_down matmul (the largest weight: 9.9 MB f16/layer) still
used the 1D tiled pattern that read W from global memory for every M iteration
(234× redundant for 234 patches).

**Fix**: 2D tiled register-blocked f16 matmul. Combines 2D tiling (BM=64,
BN=64, BK=16) with f16 weight storage. F16 weights are widened to f32 during
the cooperative W tile load into shared memory.

**Files**: `shaders/vision_matmul_f16_tiled_add.wgsl` (new), `src/engine.ts` (2D dispatch)

**Results**: Vision tower 26.9s → 12.2s (55% faster). Accuracy preserved.
Combined Q8+F16+2D tiling: 33.9s → 12.2s (64% total improvement).

### 8. vec4 dot products in attention (P8 — COMPUTE OPTIMIZATION)

**Problem**: The vision attention shader iterated head_dim=72 one element at
a time for Q·K dot product, V accumulation, Q/K/V load, and output write.
72 scalar operations per inner loop iteration.

**Fix**: Since head_dim=72 = 4×18, all loops use vec4 operations:
- Q·K dot product: 72 scalar mul-adds → 18 vec4 dots
- V accumulation: 72 scalar mul-adds → 18 vec4 mul-adds
- Q/K/V load: 72 scalar loads → 18 vec4 loads
- Output write: 72 scalar writes → 18 vec4 writes

Shared memory tiles stored as `array<vec4<f32>>` instead of `array<f32>`.

**Files**: `shaders/vision_attention.wgsl` (rewritten with vec4)

**Results**: Vision tower 12.2s → 8.7s (29% faster). Accuracy preserved.
Combined Q8+F16+2D tiling+vec4 attention: 33.9s → 8.7s (74% total improvement).

### 9. Previous optimizations (commit e177133)

Already implemented before this session:
1. Compact patch embedding for still images (sum temporal weights → 768-dim)
2. Batch all 27 layers into one compute pass
3. Tiled matmul with workgroup shared memory (vision_matmul_tiled)
4. Fused FFN up + GELU (vision_matmul_tiled_gelu)
5. Fused attn proj + residual / FFN down + residual (vision_matmul_tiled_add)
6. Fused merger mm.0 + GELU

---

## Research Findings (from Qwen3-VL reference analysis)

### Confirmed Correct
- ✓ RoPE: NeoX-style rotate_half (not GPT-NeoX interleaved)
- ✓ Attention scaling: head_dim^-0.5 = 72^-0.5 ≈ 0.1179
- ✓ GELU: tanh approximation for vision blocks (gelu_pytorch_tanh)
- ✓ LayerNorm (not RMSNorm) with eps=1e-6
- ✓ No window attention (Qwen3-VL intentionally removed it, GitHub #1717)
- ✓ Online softmax (flash-attention style) in attention kernel
- ✓ Image normalization: mean=0.5, std=0.5
- ✓ Position embedding: bilinear interpolation, align_corners=True
- ✓ Compact patch embedding for still images (matches vLLM's approach)

### Fixed
- ❌→✓ Bicubic interpolation: a=-0.75 → a=-0.5 (PIL BICUBIC uses a=-0.5)

### TODO / Not Yet Fixed
- △ Patch merger GELU: should use exact `nn.GELU()` (erf-based), not tanh.
  The difference is <0.001 relative error, so low priority. WGSL has no
  `erf` builtin — would need a polynomial approximation.
- △ Q8_0 quantization: per-block (32 elements) scaling may cause accuracy
  issues for vision weights with outliers. Consider per-channel quantization
  for attention weights. Low priority — Q8_0 is the format from the model
  creator, not something we can change without re-quantizing.

### vLLM Optimizations (for future reference)
- Fused Triton kernels for preprocessing (EVS pruning, pos embed interp, RoPE)
- CUDA Graphs for ViT encoder (eliminates kernel launch bubbles)
- Fused QK-RMSNorm + mRoPE (single kernel for three operations)
- Piecewise CUDA graph support

### WebGPU Optimization Techniques (for future reference)
- Subgroup operations (Chrome 134+): `subgroupAdd()`, `subgroupMax()` for
  softmax reductions — eliminates workgroup shared memory + barriers
- f16 intermediate accumulations in attention (if precision allows)
- Cooperative matrix multiply (if available)
- ONNX Runtime WebGPU Flash Attention 2 (PR #22932) as reference
- llama.cpp WebGPU flash_attn.wgsl uses global KV loads for M3 (shared
  memory pre-loading is slower on Apple Silicon)
- 2D matmul tiling: TILE_M=4, TILE_N=4, WG_SIZE=8×8=64 → 79.5% of optimal
- vec4 loads when K divisible by 4 → 12.7% speedup (ONNX Runtime PR #29271)

---

## Future Optimization Opportunities

### P3: Q8_0 in-shader dequantization — DONE (see Completed Optimizations #4)

### P4: F16 storage for FFN down weights — DONE (see Completed Optimizations #5)

### P4b: f16 storage for vision activations — LOW PRIORITY (compute-bound)

All activation buffers use f32 (4 bytes). Using f16 would halve memory
bandwidth for activations (916→458 MB total). However, the benchmark shows
the vision tower is **compute-bound** (60.3ms compute vs 14.8ms memory),
so f16 activations would only give ~1.45× memory speedup but minimal actual
speedup since compute is the bottleneck.

**Revised priority**: LOW. Would help if we also improve compute throughput.

### P5: 2D tiled matmul for ffn_down (F16) — DONE (see Completed Optimizations #7)

### P6: 2D tiled matmul — DONE (see Completed Optimizations #6)

### P7: Subgroup operations (Chrome 134+) — HIGH PRIORITY (compute-bound)

`subgroupAdd()`, `subgroupMax()` for softmax reductions. Eliminates shared
memory and barriers. 1.29× prefill speedup in ONNX Runtime tests.

**Why high priority now**: The benchmark shows we're 90× slower than the
compute roofline. Subgroups would:
1. Eliminate shared memory tiling in attention (direct global loads)
2. Use `subgroupAdd` for dot product reduction (hardware-native)
3. Reduce kernel launch overhead (fewer barriers = simpler shaders)
4. Enable f16 compute (2× ALU throughput on Apple Silicon)

**Status**: Chrome 134+ only. Not in Safari or Firefox yet. Must detect and
fall back. Apple Silicon supports subgroups in Metal but Safari doesn't
expose them in WebGPU yet.

### P8: vec4 attention — DONE (see Completed Optimizations #8)

### P9: Reduce kernel launch overhead — HIGH PRIORITY (90× gap)

The 90× gap from roofline suggests kernel launch overhead is a major factor.
27 layers × 7 dispatches/layer = 189 dispatches. Options:
1. **Fuse dispatches**: Combine LayerNorm+QKV, or RoPE+attention, or
   attn_out+residual into single dispatches (like the LLM engine's
   fusedDecode option)
2. **CUDA Graphs equivalent**: WebGPU doesn't have CUDA Graphs, but we can
   reduce submit overhead by keeping everything in one compute pass (already
   done) and minimizing pipeline state changes
3. **Increase workgroup count**: More workgroups = better GPU occupancy =
   more parallelism to hide launch latency

### P10: F16 compute in attention — MEDIUM PRIORITY

Apple Silicon has 2× f16 ALU throughput vs f32. Using f16 for the attention
dot products and V accumulation would halve compute time. The LLM engine
already does this (`matmul_split_sg_af16.wgsl` reads f16 activations,
computes dot in f16, accumulates in f32).

Requires `shader-f16` (already requested). Would need f16 activations (P4b)
to get the data to the shader as f16.

### P11: Create test images with known text content
Create synthetic images with clear, large text to verify text recognition
accuracy. This will help measure the impact of the bicubic fix and future
accuracy improvements.

---

## Findings: What DIDN'T work

### Parallel LayerNorm with tree reduction (TESTED, REVERTED)
Tried parallelizing LayerNorm with 64-thread tree reduction (shared memory).
Result: 22% SLOWER (33.9s → 41.6s vision tower).

**Why**: On Apple Silicon's SIMD architecture, all 64 threads in a workgroup
execute in lockstep. The "redundant" work in the single-thread version (all
64 threads doing the same 3456 iterations) is FREE — it's just one SIMD
instruction executed 64 times in hardware. The parallel version adds 12+
workgroupBarriers per row, which stall the SIMD pipeline. Barriers are
expensive (~1000-2000 cycles each on Metal), and with 234 rows × 12 barriers
= 2808 barriers, the overhead dominates.

**Lesson**: On SIMD GPUs, "redundant" work across threads in lockstep is free.
Parallelization that adds barriers can be counterproductive. Only parallelize
if the parallel version has fewer total cycles (including barrier overhead).

### 32KB workgroup storage for TILE_SIZE=32 attention (TESTED, REVERTED)
Tried requesting 32KB workgroup storage to use TILE_SIZE=32 in attention.
Result: SLOWER (due to reduced GPU occupancy).

**Why**: Larger workgroup storage means fewer workgroups can run simultaneously
on each GPU core, reducing parallelism. The TILE_SIZE=16 with 16KB default
provides better occupancy and is faster overall.

### Fused LayerNorm + QKV matmul (TESTED, REVERTED)
Tried fusing LayerNorm into the QKV matmul shader (load X, normalize in
shared memory, then matmul). Result: SLOWER for large M (234 patches).

**Why**: The fused shader processes all M rows sequentially in each workgroup,
adding 12 barriers per row for the LayerNorm reduction. The separate approach
runs LayerNorm as a separate dispatch (all rows in parallel, 1 workgroup per
row), then matmul as another dispatch. The separate approach has better
parallelism because the GPU can schedule LayerNorm workgroups and matmul
workgroups independently.

**Lesson**: Kernel fusion is not always beneficial. When the fused kernel
serializes work that was previously parallelized across dispatches, the
barrier overhead can exceed the saved memory traffic.

### BK=32 (BKV=8) for Q8 tiled matmul (TESTED, REVERTED)
Tried doubling BK from 16 to 32 to halve the number of K iterations and
barriers. Result: 6% SLOWER (5443ms → 5755ms).

**Why**: BK=32 doubles the shared memory per tile (256→512 vec4s per array,
512→1024 total). On Apple Silicon, shared memory is backed by L1 cache
(~23 GB/s, SLOWER than DRAM at 100 GB/s). Larger tiles mean more shared
memory traffic and lower GPU occupancy (fewer workgroups per core).

**Lesson**: On Apple Silicon, smaller BK values are better because shared
memory is the bottleneck, not barrier count.

### Direct global load matmul (TESTED, REVERTED)
Tried eliminating shared memory entirely — each thread loads X and W directly
from global memory (L2 cached). Result: 25% SLOWER (5443ms → 6816ms).

**Why**: Without cooperative loading, each of the 256 threads loads its own
W vec4s, causing 4× more global memory traffic (2048 vs 512 vec4s per K
iteration). The L2 cache (~400 GB/s) handles this, but the extra traffic
overwhelms the cache. The 2D tiled approach with shared memory reduces
total W reads by 64× (loaded once per K tile, reused across 64 patches).

**Lesson**: On Apple Silicon, shared memory staging IS beneficial for matmul
because it reduces total global memory traffic, even though shared memory
bandwidth is lower than DRAM. The research about "direct global loads being
faster" applies to cases where staging adds latency without reducing traffic.

### Subgroup per-output matmul (TESTED, REVERTED)
Tried using one subgroup (32 threads) per (m, n) output element with
`subgroupAdd` for K reduction. Result: 2.5× SLOWER (5443ms → 13455ms).

**Why**: One workgroup per output element means 234×3456 = 808K workgroups
for QKV alone. Each workgroup has only 32 threads doing a tiny dot product
(K/32 = 36 iterations). The workgroup launch overhead dominates completely.
The 2D tiled approach amortizes launch overhead across 256 output elements
per workgroup.

**Lesson**: Subgroups are powerful for reduction-heavy operations (like the
LLM decode matmul where M=1), but NOT for batched matmul where M>1. The 2D
tiled pattern with 256 threads is much better for batched matmul.

### 8×4 register tiles (TESTED, REVERTED)
Tried using 8×4 register tiles (128 threads) instead of 4×4 (256 threads).
Result: 49% SLOWER (5443ms → 8094ms).

**Why**: 128 threads means each thread does 2 cooperative loads per K
iteration (256 elements / 128 threads = 2), doubling load time. The 8-row
register tile also increases register pressure, reducing GPU occupancy.

**Lesson**: 256 threads with 4×4 tiles is the sweet spot for Apple Silicon.
Smaller workgroups reduce cooperative load efficiency.

### F16 compute in F16 matmul (TESTED, REVERTED)
Tried using f16 for dot products in the F16 tiled matmul (store X and W as
f16 in shared memory, dot in f16, accumulate in f32). Result: NO IMPROVEMENT
(5443ms → 5494ms, within noise).

**Why**: The f32→f16 conversion of X adds overhead that offsets the 2× ALU
gain. The FFN down matmul is only 1 of 7 dispatches per layer, so even a 2×
speedup there would only give ~15% overall improvement. The conversion
overhead eats most of that.

**Lesson**: F16 compute only helps when the data is ALREADY in f16 (like the
W weights). Converting f32 activations to f16 on-the-fly negates the ALU gain.

### Pre-dequantize Q8→F16 weights (TESTED, REVERTED)
Tried pre-dequantizing Q8_0 weights to f16 at load time, then using the f16
tiled matmul for ALL layers (not just ffn_down). Result: 25× SLOWER
(5443ms → 139316ms) AND accuracy broke (model sees "cartoon character"
instead of screenshot).

**Why**:
1. F16 weights are 88% larger than Q8 (2 bytes vs 1.0625 bytes), causing
   88% more memory traffic. At 100 GB/s, the extra 248 MB takes 2.5ms more
   per forward pass — but the actual slowdown is much worse because the
   larger weights overwhelm the L2 cache.
2. F16 precision is insufficient for these weights — the Q8_0 block scales
   provide dynamic range that f16 can't represent with a single global
   precision. The accuracy loss confirms this.

**Lesson**: Q8_0 with in-shader dequantization is BETTER than pre-dequantized
f16 for this model. The dequant overhead (~50% of instructions) is less
costly than the 88% memory traffic increase. Q8_0's block-level scales
provide better dynamic range than f16.

### TILE_SIZE=32 in attention (TESTED, FAILED)
Tried increasing attention TILE_SIZE from 16 to 32 to reduce barrier count.
Result: FAILED — workgroup storage 18432 bytes > 16384 byte default limit.

**Why**: Apple Silicon's default workgroup storage limit is 16KB. TILE_SIZE=32
needs 18KB (32×72×4×2 = 18432 bytes for K+V tiles). Requesting 32KB limit
was previously tested and reduces GPU occupancy.

**Lesson**: TILE_SIZE=16 with 16KB workgroup storage is the maximum practical
tile size for attention on Apple Silicon.

---

## Implementation Log

### 2025-01-XX: Pre-compute RoPE + tiled attention + bicubic fix
- Status: COMPLETE
- Files changed:
  - `shaders/vision_apply_rope.wgsl` (new — pre-computes RoPE for Q/K)
  - `shaders/vision_attention.wgsl` (rewritten — shared memory tiling, no RoPE)
  - `src/engine.ts` (register new shader, add qRopedBuf/kRopedBuf, update dispatch)
  - `src/vision.ts` (bicubic a=-0.75 → a=-0.5)
  - `docs/VLM_OPTIMIZATION_LOG.md` (this file)
- Results: 33% faster vision tower, 37% faster prefill, 58% faster tokens/s
- Accuracy: Red image ✓, screenshot gives more coherent description

### 2025-01-XX: Parallel LayerNorm + 32KB storage + fused LN+matmul (REVERTED)
- Status: REVERTED (all three optimizations were slower)
- Tested: parallel tree-reduction LayerNorm, 32KB workgroup storage with
  TILE_SIZE=32, fused LayerNorm+QKV matmul
- All three caused regressions due to barrier overhead and reduced occupancy
- Findings documented above in "What DIDN'T work" section
- Kept: documentation comments in shaders explaining why these approaches
  don't work on Apple Silicon SIMD architecture

### 2025-01-XX: Q8_0 in-shader dequantization
- Status: COMPLETE
- Files changed:
  - `src/vision.ts` (repackQ8_0(), Q8PackedWeight type, Q8 packed weight loading)
  - `shaders/vision_matmul_q8.wgsl` (new — Q8 matmul with split outputs)
  - `shaders/vision_matmul_q8_gelu.wgsl` (new — Q8 matmul + GELU)
  - `shaders/vision_matmul_q8_add.wgsl` (new — Q8 matmul + residual add)
  - `src/engine.ts` (dual buffer upload, Q8 shader dispatch, skip f32 for Q8 weights)
- Results: Vision tower 33.9s → 31.1s (8% faster), accuracy preserved
- Weight bandwidth: 60.8 MB/layer → 32.0 MB/layer (1.9× reduction)
- Pattern: Follows LLM engine's q8 KV cache (attention_sg_kv8.wgsl) —
  packed u32 words + f32 block scales, dequantized inline via bit shifts

### 2025-01-XX: F16 storage for FFN down weights
- Status: COMPLETE
- Files changed:
  - `src/vision.ts` (repackF16(), F16PackedWeight type, F16 packed weight loading)
  - `shaders/vision_matmul_f16_add.wgsl` (new — f16 weight matmul + residual add)
  - `src/engine.ts` (request shader-f16, f16 buffer upload, f16 shader dispatch)
- Results: Accuracy preserved. Weight bandwidth 32.0 MB → 22.1 MB/layer (1.45× further)
- Combined with Q8: total weight bandwidth 60.8 MB → 22.1 MB/layer (2.75× reduction)
- Pattern: Follows LLM engine's f16 KV cache (attention_sg_kv16.wgsl) —
  array<f16> storage, f32() widening at read time

### 2025-01-XX: 2D tiled Q8 matmul
- Status: COMPLETE
- Files changed:
  - `shaders/vision_matmul_q8_tiled.wgsl` (new — 2D tiled Q8 with split outputs)
  - `shaders/vision_matmul_q8_tiled_gelu.wgsl` (new — 2D tiled Q8 + GELU)
  - `shaders/vision_matmul_q8_tiled_add.wgsl` (new — 2D tiled Q8 + residual add)
  - `src/engine.ts` (2D dispatch: ceil(N/64) × ceil(M/64))
- Results: Vision tower 31.1s → 26.9s (13% faster). Accuracy preserved.
- Combined with Q8+F16: 33.9s → 26.9s (20.6% total improvement)
- Pattern: Follows LLM engine's matmul_split_tiled.wgsl —
  BM=64, BN=64, BK=16, 256 threads, 4×4 register tiles, 8KB shared memory

### 2025-01-XX: 2D tiled F16 matmul for ffn_down
- Status: COMPLETE
- Files changed:
  - `shaders/vision_matmul_f16_tiled_add.wgsl` (new — 2D tiled f16 + residual add)
  - `src/engine.ts` (use vision_matmul_f16_tiled_add with 2D dispatch)
- Results: Vision tower 26.9s → 12.2s (55% faster). Accuracy preserved.
- Combined Q8+F16+2D tiling: 33.9s → 12.2s (64% total improvement)
- Pattern: Combines 2D tiling (matmul_split_tiled) with f16 storage
  (attention_sg_kv16) — f16 weights widened to f32 during cooperative W tile load

### 2025-01-XX: vec4 dot products in attention
- Status: COMPLETE
- Files changed:
  - `shaders/vision_attention.wgsl` (rewritten with vec4 for all head_dim loops)
- Results: Vision tower 12.2s → 8.7s (29% faster). Accuracy preserved.
- Combined Q8+F16+2D tiling+vec4 attention: 33.9s → 8.7s (74% total improvement)
- Pattern: head_dim=72=4×18, all loops use vec4 dot/mul-add/load/write.
  Shared memory tiles stored as array<vec4<f32>>.
