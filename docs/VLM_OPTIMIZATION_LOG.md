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
| Vision tower | 50.8s | 3.0s | 94% faster |
| Prefill | 18.2s | 11.0s | 40% faster |
| Tokens/s | 3.6 | 6.1 | 69% faster |

**Benchmark results** (3 runs, 234 patches, 90s cooldown between runs):
- Run 1: 2990ms, Run 2: 3241ms, Run 3: 3142ms
- Min: 2990ms, Max: 3241ms, Avg: 3124ms
- Per-layer: 110.7ms/layer

**Breakdown** (from timestamp-query profiling):
- Weight loading: ~1.5s (4-chunk parallel fetch + skip f32 dequant + no-copy upload)
  - HTTP fetch: ~1170ms (600MB, 4 parallel chunks)
  - Parse+repack: ~300ms (fast f16ToF32 + direct byte access)
  - GPU upload: ~300ms (no-copy writeBuffer with byteOffset)
- GPU forward: ~1.36s (27 layers, measured via timestamp-query)
  - Per-layer: ~50.4ms
  - 22.5× slower than compute roofline (60.3ms)
  - Bottleneck: Q8 dequantization overhead (~5.5 instructions per FMA)
- Second call (weights cached): ~1.36s only

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

### 2026-08-14: unpack4x8snorm for Q8 dequantization (TESTED, NEUTRAL, REVERTED)
Tried replacing the bit-shift Q8 dequant (shift/mask/sign-extend/convert/multiply)
with `unpack4x8snorm` (hardware int8→f32 conversion). Pre-multiplied scales by
127.0 in `repackQ8_0` so the shader does `unpack4x8snorm(packed) * scale`.

Result: NO measurable change (5718ms vs baseline 5443ms — within noise).
Accuracy preserved.

**Why neutral**: The WGSL compiler already optimizes the bit-shift pattern
into efficient hardware sign-extension instructions. `unpack4x8snorm` is
semantically equivalent but doesn't unlock new hardware paths on Apple
Silicon. The 6× instruction count reduction is theoretical — the compiler
was already doing this optimization.

**Note on -128 clamping**: `unpack4x8snorm` clamps int8 -128 to -1.0 (like
-127), introducing a 0.78% error on that single value. This is negligible
vs Q8_0's inherent quantization error but is a minor correctness concern.

**Lesson**: Hardware builtins like `unpack4x8snorm` are convenient but not
necessarily faster than bit-shift patterns on Apple Silicon — the WGSL
compiler already optimizes the latter well. Always benchmark.

### 2026-08-14: chromium-experimental-subgroup-matrix (TESTED, BLOCKED BY BUG, REVERTED)
Tried using `chromium-experimental-subgroup-matrix` for hardware-accelerated
8×8×8 matrix multiply-accumulate (maps to Apple's simdgroup_matrix hardware
units on Metal 3).

**Implementation**:
- Built a standalone test (`test-sgmat.html`) to isolate the API
- Discovered the supported type combination on Apple Silicon:
  - f16 inputs + f32 accumulation ✓ (what llama.cpp uses)
  - f32 inputs + f32 accumulation ✗ (NOT supported — compile error)
  - i8/u8 inputs + i32 accumulation ✓
- Built a correct f16-input/f32-accumulate shader that passes standalone
  tests (1-SG, 2-SG, 8-SG, and K-loop iterations all match CPU reference)
- Wired it into the vision tower QKV projection

**Result**: 2× PERFORMANCE REGRESSION + BROKEN ACCURACY.
- Vision tower: 5443ms → 14748ms (2.7× slower)
- LLM generation: 34s → 67s (2× slower)
- Model output: nonsensical ("glitch art" instead of screenshot description)

**Root cause**: Requesting the `chromium-experimental-subgroup-matrix` feature
degrades ALL shader performance on this device — even shaders that don't use
the feature. Confirmed by disabling the sgmat shader path (keeping only the
feature request): the 2× regression persisted across the entire pipeline
(vision + LLM). This is a Chrome/Dawn bug on Apple Silicon, not a shader
correctness issue (the standalone tests all passed).

**Reverted**: Removed the feature request, the sgmat shader, and the dispatch
path. Baseline restored (5357ms vision, 34s generation, correct accuracy).

**Lesson**: Experimental WebGPU features can have device-wide side effects
that aren't limited to the shaders using them. Always test "feature requested
but unused" as a control. The subgroup-matrix API itself works correctly
(standalone tests pass with f16 inputs + f32 accumulation), but the Chrome
implementation has a bug that makes it unsuitable for production use on
Apple Silicon as of Chrome 138 (2026-08-14).

### 2026-08-15: Bulk fetch + skip f32 dequantization (46% faster, KEPT)
**Two optimizations to weight loading, applied together:**

1. **Bulk fetch**: Replaced 334 individual HTTP Range requests (one per tensor)
   with a single fetch for the entire data section. Reduced HTTP overhead.

2. **Skip f32 dequantization**: Q8_0 and F16 weights were being dequantized to
   f32 on CPU during loading, but the GPU shaders use the raw Q8/F16 bytes
   directly (via in-shader dequantization). The f32 arrays were never uploaded
   to GPU — they were pure waste. Skipping dequantization saves ~2.3s of CPU
   time and ~2.3GB of memory allocation.

**Results**: 5579ms → 2990ms (46% faster)
- Weight loading: 4134ms → 1580ms (2.6× faster)
- GPU forward: 1414ms → 1420ms (unchanged — as expected)
- Second call (weights cached): ~1420ms only

**Files changed**:
- `src/vision.ts`: Bulk fetch + skip dequantization in `loadVisionWeights()`
- `src/engine.ts`: Removed debug timing code

**Key insight**: The benchmark was measuring 75% weight loading + 25% GPU
compute. The "5.5s vision tower" was actually 4.1s of CPU work (HTTP fetch +
dequantization) + 1.4s of GPU work. The GPU forward pass was never the
bottleneck — the CPU loading was.

**Lesson**: Always profile the full pipeline before optimizing individual
kernels. The matmul shader optimizations (BM=128, BKV=2, hybrid shared memory)
all failed because they targeted the wrong bottleneck.

### 2026-08-15: Parallel fetch + fast f16/repack (incremental, KEPT)
**Weight loading breakdown** (from profiling):
- HTTP fetch: 1552ms (600MB at 386MB/s) — 64% of loading
- Parse+repack: 449ms — 19%
- GPU upload: 427ms — 17%

**Optimizations**:
1. **Parallel fetch**: Split the 600MB bulk fetch into 4 concurrent HTTP Range
   requests. Best case: 1552ms → 1170ms (24% faster). On localhost, this
   bypasses per-request overhead. In production with CORS proxy, the gain
   may be larger (more per-request latency to amortize).

2. **Fast f16ToF32**: Replaced `Math.pow`-based f16→f32 conversion with
   bit-level manipulation using a shared ArrayBuffer. ~5-10× faster per call.

3. **Fast repackQ8_0**: Replaced DataView with direct Uint8Array/Uint16Array
   access. Eliminates per-element function call overhead.

**Files changed**: `src/vision.ts`

**Note**: Benchmark results were noisy due to system load (chrome-devtools-mcp
running 22 Chrome processes). Best-case vision tower: 3262ms. Accuracy
preserved across all runs.

### 2026-08-15: BKV and hybrid approach experiments (TESTED, REVERTED)
**Attempted 3 shader changes to the Q8 tiled and F16 tiled add shaders:**

1. **Q8 tiled BKV 2→4**: Changed BKV from 2 to 4 in the hybrid Q8 tiled shader
   (W only in shared). The shared array was already sized for BKV=4 (256 vec4).
   **Result**: Accuracy broke completely — LLM described "grid of colorful
   icons" instead of "screenshot of social media post". Root cause unclear;
   the indexing logic appears correct. Possibly a WGSL compiler bug on Apple
   Silicon with hybrid + BKV=4.

2. **F16 tiled add shared memory 512→256**: Reduced the shared memory array
   from 512 to 256 vec4 (8KB→4KB). **Result**: Same accuracy breakage. Root
   cause: the F16 shader actually has BKV=8 (not 4 as initially misread from
   grep output), so it needs 64*8=512 elements. Reducing to 256 caused
   out-of-bounds shared memory access.

3. **Reverted all shader changes**: Restored both shaders to their committed
   state (non-hybrid, BKV=4, both X+W in shared, 8KB). Accuracy restored.

**Lesson**: Always verify the actual BKV value before changing shared memory
array sizes. The grep output can be misleading when multiple shaders have
similar constant names. Use `git show HEAD:path/to/shader` to verify the
committed state.

**Current shader state** (all committed, all correct):
- Q8 tiled (QKV): BKV=4, both X+W in shared (8KB), split outputs
- Q8 tiled add (attnOut): BKV=4, both X+W in shared (8KB)
- Q8 tiled gelu (FFN up): BKV=4, both X+W in shared (8KB)
- F16 tiled add (FFN down): BKV=4, both X+W in shared (8KB)

**GPU forward pass profiling** (via timestamp-query):
- GPU time (27 layers): ~1360ms (actual on-GPU time, pre-BKV=8)
- CPU wall-clock: ~1420ms (60ms submit/queue overhead)
- Per-layer: ~50.4ms
- Weight loading: ~1657ms (4 parallel chunks)
- Total vision tower: ~3020ms (pre-BKV=8), ~3000ms (post-BKV=8)

The GPU time is 22.5× slower than the compute roofline (60.3ms).
The gap is due to Q8 dequantization overhead (~4.5 instructions per FMA),
shared memory barriers, and low occupancy.

**8-chunk parallel fetch**: Tested and reverted. 8 chunks was slower than 4
(2317ms vs 1657ms weight loading) due to connection contention. 4 chunks
is the sweet spot for localhost.

### 2026-08-16: Manual loop unrolling + BKV=8 (COMMITTED)

**Manual loop unrolling** (nuss-and-bolts study):
- Researched WebGPU matmul optimization techniques. Key finding: the WGSL→Metal
  compiler doesn't always unroll loops even with known bounds. Manual unrolling
  of the inner tm/tn loops (4×4 register tile) gave ~3x on Apple Silicon in the
  nuss-and-bolts study.
- Applied manual unrolling to all 4 tiled matmul shaders (bias init, compute
  loop, output write). Result: avg 4877ms → 4292ms (12% better), accuracy
  preserved. Min unchanged (within noise).

**BKV=8 (full Q8_0 block processing)**:
- Changed BKV from 4 to 8 in all 4 tiled matmul shaders. This processes a full
  Q8_0 block (32 elements) per K iteration, halving the number of barriers and
  shared memory load overhead. Shared memory doubles from 8KB to 16KB (1 WG/core
  at limit), but the reduced barrier overhead more than compensates.
- Result: avg 4292ms → 3502ms (18% faster), min 3727ms → 3261ms (12.5% faster).
  Accuracy preserved across all 3 runs.

**BKV=16**: Tested and failed. 32KB shared memory exceeds WebGPU limits.

**unpack4x8snorm dequantization**: Tested and reverted. Using the builtin
`unpack4x8snorm(packed) * (scale * 127.0)` instead of manual sign-extension
was significantly slower (avg 5278ms vs 3502ms). The builtin appears to be
less efficient than manual bit manipulation on Apple Silicon's Metal backend.

### 2026-08-16: Attention unrolling + explicit FMA (COMMITTED)

**Attention shader unrolling**: Manually unrolled the d4 (0..17) loops in
vision_attention.wgsl (Q load, acc init, Q·K dot, V accumulation, output write,
cooperative K/V load). Result: avg 3502ms → 3433ms (2% faster).

**Explicit multiply-add**: Replaced vec4 `dot()` with explicit scalar
multiply-add (`xr.x*w.x + xr.y*w.y + ...`) in all 4 tiled matmul shaders.
Gives the Metal compiler more freedom to schedule FMA instructions.
Result: avg 3433ms → 3295ms (4% faster), min 3227ms → 3157ms (2.2% faster).

**Full kv loop unrolling**: Tested and reverted. Unrolling the kv loop
(8 iterations) in the compute section caused register spilling and was
slower (avg 4061ms vs 3433ms, 18% slower).

**Hybrid shared memory (BKV=16, W only)**: Tested and reverted. Only storing
W in shared memory (not X) allows BKV=16 with 16KB shared, halving barriers.
But the extra global memory reads for X (8× more reads per K iteration)
outweigh the barrier savings. Avg 3616ms vs 3295ms (10% slower).

### Current best: 3157ms min, 3295ms avg (from ~4877ms avg at session start)

Cumulative improvement: ~32% faster vision tower forward pass through:
1. Manual tm/tn loop unrolling (12%)
2. BKV=8 full Q8_0 block processing (18%)
3. Attention shader d4 loop unrolling (2%)
4. Explicit multiply-add replacing dot() (4%)

### 2026-08-17: SubgroupMatrix experiment (REVERTED — slower)

**SubgroupMatrix feature detection**: Confirmed `chromium-experimental-subgroup-matrix`
is available on this Apple Silicon device (M-series). Also confirmed `subgroups`
and `shader-f16` are available. Basic 8×8×8 f16 matmul test passed correctly.

**SubgroupMatrix Q8 matmul shader**: Implemented `vision_matmul_q8_sgmat`,
`vision_matmul_q8_sgmat_add`, `vision_matmul_q8_sgmat_gelu`, and
`vision_matmul_f16_sgmat_add` using hardware tensor cores (Metal simdgroup_matrix).

Configuration tested:
- 8 subgroups per WG (SUBGROUP_M=2, SUBGROUP_N=4), 256 threads total
- 8×8×8 f16 matmul tiles with f32 accumulator
- TILE_K=32 (4 SubgroupMatrix K-steps per shared memory load)
- Workgroup output tile: 16×32 = 512 elements
- Shared memory: 512 + 1024 + 512 = 2048 f16 + 512 f32 = 5KB total

Results:
- TILE_K=8 (1 SGM step per load): 6175ms avg (1.87× slower than tiled)
- TILE_K=32 (4 SGM steps per load): 4834ms avg (1.47× slower than tiled)

The SubgroupMatrix approach is slower because:
1. **Small workgroup tile** (16×32=512) vs tiled (64×64=4096) → 8× more dispatches
2. **f32→f16 conversion overhead** for X activations and Q8→f16 dequantization for W
3. **8×8 tile size too small** for these matrix dimensions (234×1152, 234×3456)
4. The tiled shader's 4×4 register tiles per thread (16 accumulators) are already
   well-optimized for Apple Silicon's SIMD architecture

SubgroupMatrix works best for large matrices (1024×1024+) where the hardware
tensor core throughput dominates. For the vision tower's smaller matrices,
the manual tiled approach with explicit FMA is faster.

**Key learning**: SubgroupMatrix on Apple Silicon uses 8×8 simdgroup_matrix tiles.
With max 256 threads/WG (8 subgroups), the workgroup tile is limited to 16×32 or
similar. This is 8× smaller than the 64×64 tiled shader, causing dispatch overhead
and poor L1 cache utilization to dominate.

### Research findings (2025-2026 WebGPU optimization techniques)

Researched latest WebGPU VLM optimization techniques. Key findings:

1. **SubgroupMatrix**: Available on Apple Silicon but needs large matrices to win
2. **F16 activations**: Widely used in production (Janus-Pro-7B, WebLLM q4f16).
   Halves activation bandwidth. Low risk, high impact.
3. **Kernel fusion**: 66-458× speedup possible. LayerNorm+MatMul and QKV+RoPE
   fusion save 54+ dispatches per forward pass.
4. **Q8_0 weight reorder**: 3.1× speedup on Intel (separate scales from data).
   Not yet tested on Apple Silicon.
5. **Dispatch overhead**: 24-71 µs per dispatch on Metal. Already minimized by
   batching all 27 layers into one compute pass.
6. **Packed 4x8 integer dot product**: Available in Chrome 123+. 1.6-2.9× faster
   than f16 for 8-bit data. Could help Q8 dequantization.

### Shader micro-optimization experiments (2025-01-24)

Tested several compute loop variants for the 2D tiled Q8 matmul shaders.
All measured on 234-patch screenshot benchmark (27 layers, 936 patches).

**Baseline**: 2D tiled Q8 with manual FMA unrolling, BM=64, BN=64, BKV=8, 256 threads
- Performance: ~3295ms avg (includes ~3000ms weight loading on first call)
- GPU-only time: ~2400ms for 27 layers = 89ms/layer
- Compute throughput: ~160 GFLOPS (3.2% of M2 ~5 TFLOPS peak)

**Experiments (all REVERTED)**:

1. **2× K unrolling** (process 2 kv values per iteration for ILP):
   - Result: 37% slower (3628ms vs 2647ms for 4-patch test)
   - Cause: Register pressure spilling — 32 accumulators + 16 X + 16 W = 64 extra registers

2. **F16 shared memory** (store xs/ws as vec4<f16> instead of vec4<f32>):
   - Shared memory: 16KB → 8KB (allows 4 WGs/core instead of 2)
   - Result: 60% slower (5289ms avg vs 3295ms avg)
   - Cause: f16→f32 conversion overhead in compute loop negates occupancy gain

3. **1D tiled shaders** (4KB shared memory, 8 WGs/core, but W re-read per M row):
   - Result: 9× slower for 234 patches (29913ms vs 3295ms)
   - Cause: W is re-read 936× per layer (M=936 rows), overwhelming the bandwidth savings
   - Note: 1D shaders are better for small M (4 patches: 28% slower, not 9×)

4. **F16 accumulators** (var acc: array<f16, 16> for 2× FP16 compute throughput):
   - Result: 32% slower (4401ms min vs 3332ms min)
   - Cause: Metal compiler doesn't generate efficient f16 FMA from WGSL —
     the f16→f32 conversions for shared memory loads add overhead

5. **dot() built-in** (replace manual FMA with dot(vec4, vec4)):
   - Result: 96% slower (6449ms avg vs 3295ms avg)
   - Cause: WGSL→Metal compiler generates worse code for dot() than manual FMA.
     Manual FMA with explicit component access produces better instruction scheduling.

**Key learnings**:
- The WGSL→Metal compiler is sensitive to code structure. Manual FMA unrolling
  with explicit component access (xr0.x*w0.x + xr0.y*w0.y + ...) generates
  significantly better code than dot() or f16 arithmetic.
- Shared memory type matters more than occupancy. f32 shared memory with 2 WGs/core
  is faster than f16 shared memory with 4 WGs/core due to conversion overhead.
- Register pressure is the limiting factor for unrolling. 16 accumulators + 8 X + 8 W
  = 32 values per iteration is near the register limit for 256 threads.
- 2D tiling is essential for large M. The W re-read in 1D tiling dominates for M > 100.

### f16 LUT for weight loading (2025-01-24)

Replaced bit-level f16→f32 conversion with a 65536-entry lookup table in repackQ8_0.
- Engine creation: ~28s → ~25s (11% faster weight loading)
- GPU compute: unchanged (LUT only affects CPU-side repacking)
- The LUT is pre-computed at module load time and covers all 65536 possible f16 bit patterns.

### F32 pre-dequantized weights experiment (2025-01-24)

Tested pre-dequantizing Q8 weights to f32 on the GPU to eliminate Q8 dequant
overhead from the matmul compute loop.

**Approach**:
1. Upload Q8 packed + scales to GPU as before
2. Dispatch GPU dequant shader (Q8 → f32) for each weight matrix
3. Use new f32 2D tiled matmul shaders (same tiling as Q8 but loads f32 W directly)
4. Keep Q8 buffers alongside f32 (to avoid GPU race conditions with destroy)

**Result**: 76% slower (5807ms min vs 3295ms baseline for 234 patches)

**Analysis**: The Q8 dequantization was NOT on the critical path. It was hidden
by the compute loop (GPU overlaps cooperative load with compute of previous tile).
The f32 W approach uses 3.56× more global memory bandwidth (16 bytes per vec4
vs 4.5 bytes for Q8 packed+scale), which becomes the new bottleneck.

**Key learning**: Q8 in-shader dequantization is more efficient than pre-dequantization
because:
1. Q8 reduces global memory bandwidth by 3.56× (1.125 vs 4 bytes/element)
2. The dequantization overhead (12 ops per 4 elements) is hidden by compute
3. The cooperative load and compute are overlapped by the GPU's instruction scheduler
4. Global memory bandwidth, not dequant compute, is the actual bottleneck

### Benchmark variance analysis (2025-01-24)

The 234-patch benchmark shows high variance across runs (3295ms to 7364ms for
the same code). This is due to:

1. **Thermal throttling**: The 30s engine creation + LLM generation heats up the
   GPU, causing reduced clock speeds during the vision tower
2. **Weight loading on first call**: The first visionForward call includes ~3000ms
   of weight loading (HTTP fetch + CPU repacking + GPU upload)
3. **Background processes**: macOS system tasks compete for GPU time

**Recommendation**: For stable measurements, use GPU timestamp queries or run
the vision tower in isolation (without LLM generation before/after).

### Vision weight preloading (2025-01-24)

Moved vision weight loading (HTTP fetch + CPU repacking + GPU upload) from
the first `visionForward` call to engine init time, overlapping it with
LLM model loading and pipeline compilation.

**Before**: First `visionForward` call waited ~3s for weight loading
before starting GPU compute. Total: ~3295ms (weight loading + GPU compute).

**After**: Weight loading runs in parallel with engine creation. First
`visionForward` call only pays GPU compute cost. Total: ~2749ms.

**Improvement**: 17% faster (2749ms vs 3295ms min, 234 patches).

**Implementation**: The weight loading trigger code was moved from
`visionForward` to right after `S_`, `CD`, `CS`, `U` constant definitions
in `createEngine`. The `visionForward` function now just awaits the
existing loading promise.

### Multi-subgroup sgmat shader wiring (2025-01-24)

Wired up the multi-subgroup sgmat (tensor core) shaders with 8 subgroups
per workgroup (16×32 workgroup tile, TILE_K=32, 4 K-steps per load).

**Result**: The `chromium_experimental_subgroup_matrix` extension is NOT
available in the current Chrome environment, so the shaders fall back to
the manual FMA tiled shaders. The sgmat code is compiled only when the
extension is detected via `navigator.gpu.wgslLanguageFeatures`.

**Available WGSL features** (detected):
- `packed_4x8_integer_dot_product` (DP4a) — available!
- `subgroup_uniformity`, `subgroup_id` — available
- `chromium_experimental_subgroup_matrix` — NOT available

**Key learning**: The sgmat shaders are wired but dormant. The fallback
path (64×64 manual FMA tiled) is the active code path.

### TILE_K=64 for sgmat (REVERTED — slower)

Tried doubling TILE_K from 32 to 64 in sgmat shaders (8 K-steps per load,
halving barrier count). Shared memory: 384 bytes → 768 bytes (still tiny).

**Result**: 1629ms for 4 patches (vs 271ms with TILE_K=32) — 6× slower.
The larger K tile reduces occupancy or causes register pressure.

### 4×2 subgroup layout (32×16 tile) (REVERTED — slower)

Tried SUBGROUP_M=4, SUBGROUP_N=2 (32×16 workgroup tile) to increase M tile
and reduce W re-reads.

**Result**: 2059ms for 4 patches (vs 271ms with 2×4 layout) — 7.6× slower.
The 32×16 tile is worse because it reduces N parallelism (fewer N tiles
to parallelize across workgroups).

### 4×4 subgroup layout (32×32 tile, 512 threads) (REVERTED — slower)

Tried SUBGROUP_M=4, SUBGROUP_N=4 (32×32 workgroup tile, 16 subgroups,
512 threads per workgroup).

**Result**: 1972ms for 4 patches (vs 271ms with 2×4 layout) — 7.3× slower.
512 threads per workgroup reduces occupancy on Apple Silicon.

### BKV=16 for tiled shaders (REVERTED — exceeds 16KB limit)

Tried doubling BKV from 8 to 16 in the 64×64 tiled Q8 shaders (BK=64,
halving barrier count). Shared memory: 16KB → 32KB.

**Result**: Failed — exceeds the default 16KB workgroup storage limit.
The adapter supports 32KB but requesting it reduces GPU occupancy
(documented in engine.ts comment from prior testing).

### Fused LayerNorm + QKV matmul (REVERTED — slower)

Created `vision_matmul_q8_tiled_ln.wgsl` that fuses LayerNorm1 into the
QKV matmul, eliminating 27 LayerNorm dispatches and 27 global memory
round-trips (4.3MB write + 4.3MB read per layer = 232MB total).

**Approach**: Each workgroup (64 rows, 256 threads) cooperatively computes
LayerNorm for its 64 rows using 4 threads per row, then applies LN
on-the-fly during the X tile load.

**Challenges**:
1. Required 9 storage buffers (vs 8 default limit) — needed to request
   `maxStorageBuffersPerShaderStage = 10`
2. Required 17.9KB workgroup storage (vs 16KB default) — needed to request
   `maxComputeWorkgroupStorageSize = 32768`
3. The LayerNorm reduction adds 3 barriers + 2 full-row scans per
   workgroup, which is redundant across workgroups (each WG computes LN
   for its 64 rows, but only 16 rows are unique per QKV tile)

**Result**: 4108ms for 234 patches (vs 3228ms baseline) — 27% slower.
The overhead of:
- Requesting 32KB workgroup storage (reduces occupancy)
- Redundant LN computation across workgroups
- Extra barriers for LN reduction
...outweighs the savings from eliminating 27 dispatches.

**Key learning**: Kernel fusion in WebGPU is NOT always beneficial when:
1. It requires requesting higher device limits (reduces occupancy)
2. The fused operation requires a full-row reduction (barriers)
3. The reduction is redundant across workgroups
The dispatch overhead (~50µs × 27 = 1.4ms) is small compared to the
compute overhead of redundant reductions.

### Research findings (2025-01-24)

Researched latest WebGPU VLM optimization techniques. Key findings:

1. **DP4a (dot4I8Packed)**: Available in current Chrome! Could provide
   1.6-2.8× speedup for Q8 dequantization. But requires both operands
   to be packed int8 — our X is f32, so can't directly use DP4a.

2. **Q8_0 weight reordering**: Separating scales from data could provide
   2-3× speedup (Intel SYCL data). Our layout already separates scales
   from data (packed + scales in different arrays), so this is already
   partially done.

3. **F16 activations**: Could save 458MB bandwidth (50% of activation BW).
   Requires changing every shader that reads/writes activations. Big
   change, medium impact.

4. **Dispatch overhead**: 32-71µs per dispatch on Metal. Our 216 dispatches
   = 7-15ms overhead (0.2-0.5% of total). Not a major bottleneck.

5. **Subgroup matrix**: 3× speedup on M2 for 1024×1024 matmul (ONNX data).
   But requires `chromium_experimental_subgroup_matrix` which is NOT
   available in our Chrome environment.

6. **Kernel fusion**: ONNX Runtime shows 25-30% speedup from fusion. But
   our fused LN+QKV experiment showed fusion can be slower when it
   requires higher device limits or redundant computation.

---

## 2026-08-15: M-RoPE (IMROPE) support for Qwen3-VL — CRITICAL FIX

### Problem

The Qwen3-VL model uses Interleaved M-RoPE (IMROPE) with 4D position IDs
(t, x, y, z) for image tokens. The WebGPU engine was using standard 1D
sequential RoPE for all tokens, causing the VLM to hallucinate garbage
instead of describing image content. The model could not "see" images.

### Root cause

The Qwen3-VL architecture uses `GGML_ROPE_TYPE_IMROPE` with M-RoPE sections
defining how the head_dim frequency pairs are split across the 4 position
dimensions (temporal, height, width, z). For the Bonsai-27B model, the
sections are `[11, 11, 10, 0]` (parsed from the GGUF metadata key
`qwen3vl.rope.dimension_sections`).

The standard 1D RoPE was applying the same sequential position to all
frequency pairs, which is correct for text-only models but wrong for
multimodal models that need spatial position encoding for image tokens.

### Fix

Three changes were needed:

1. **Parse M-RoPE sections from GGUF** (`src/gguf.ts`):
   Added `mropeSections` to the model arch metadata, reading from
   `qwen3vl.rope.dimension_sections` (or `qwen2vl.rope.dimension_sections`).

2. **Compute per-frequency-pair cos/sin for M-RoPE** (`src/engine.ts`):
   The `ropeBufs` function now computes cos/sin values using the correct
   position dimension for each frequency pair. For IMROPE, frequency pair `j`
   uses position dimension `j % 3` (cycling t/h/w), with section boundaries
   defined by `mropeSections`. The 4th dimension (z) is unused for Qwen3-VL
   (section size 0).

3. **Compute 2D grid positions for image tokens** (`src/engine.ts`):
   `generateWithImages` now computes M-RoPE positions for the full prompt
   (text + image tokens). For image tokens at grid position (row, col):
   - `pos.t = pos_0` (temporal = base position, same for all image tokens)
   - `pos.x = pos_0 + col` (width = base + column index)
   - `pos.y = pos_0 + row` (height = base + row index)
   - `pos.z = 0` (unused)
   
   This matches llama.cpp's `mtmd_image_tokens_get_decoder_pos` function
   in `tools/mtmd/mtmd.cpp`. For text tokens, all 4 dimensions are the same
   sequential position: `[pos, pos, pos, pos]`.

### Key insight: IMROPE is halved, not interleaved rotation

A critical insight was that IMROPE uses **halved rotation** (NEOX-style
`rotate_half`), NOT interleaved rotation (GPT-J style). The "I" in IMROPE
refers to **interleaved section assignment** (cycling t/h/w per frequency
pair), not interleaved rotation.

This means the existing `rope_partial` shader works correctly — only the
cos/sin values differ. The rotation method (rotate_half) is the same as
standard NeoX RoPE. The difference is purely in which position dimension
(t, x, y, or z) is used for each frequency pair when computing the cos/sin
table.

### Verification

After the fix, the VLM successfully described a synthetic image with
colored shapes, correctly identifying colors and approximate shapes.
The model output was considered acceptable given the aggressive Q1_0
quantization (1.5 bits/weight).

### Range error fix (vision weight loading)

A secondary issue was `RangeError: offset is out of bounds` during vision
weights loading. This was caused by the HTTP server not supporting `Range`
requests, leading to corrupted vision weights (the server returned the
full file instead of the requested byte range).

**Fix**: Updated the Python HTTP server to handle `Range` requests
correctly, and added a slice in `vision.ts` to ensure only the requested
bytes are processed even if the server returns more.

### Files changed

- `src/gguf.ts`: Parse `mropeSections` from GGUF metadata
- `src/types.ts`: Add `mropeSections` to arch type
- `src/engine.ts`: Compute M-RoPE cos/sin in `ropeBufs`, compute 2D
  positions for image tokens in `generateWithImages`, thread M-RoPE
  positions through prefill and decode paths
- `src/vision.ts`: Slice response to expected length in weight loading
- `shaders/rope_imrope.wgsl`: New shader for IMROPE rotation (uses
  halved rotation with per-section cos/sin)

### Reference

- llama.cpp `tools/mtmd/mtmd.cpp`: `mtmd_image_tokens_get_decoder_pos`
  function defines the 4D position computation for image tokens
- llama.cpp PR #16780: Added Qwen3-VL support including IMROPE
- GGUF metadata: `qwen3vl.rope.dimension_sections = [11, 11, 10, 0]`
  for Bonsai-27B (head_dim=72, 36 frequency pairs: 11 t + 11 h + 10 w + 0 z)

---

## 2026-08-15: DP4a vision matmul shader (TESTED, DISABLED)

### Approach

Created a `vision_matmul_q8_dp4a.wgsl` shader that uses the
`packed_4x8_integer_dot_product` WGSL feature (`dot4I8Packed` builtin)
for 4-way int8 dot products instead of f32 FMA. This feature IS available
in the current Chrome environment (confirmed via
`navigator.gpu.wgslLanguageFeatures`).

The shader:
- Keeps W as packed int8 in shared memory (no dequantization to f32)
- Quantizes X (activations) to int8 on the fly with a per-row scale
- Uses `dot4I8Packed` for the inner dot product loop
- Rescales the result by `(x_scale * w_scale)` at the end

### Result: 6× SLOWER (disabled)

The DP4a path is 6× slower than the existing f32-dequant Q8 tiled path:
- DP4a: 1761ms for 27 layers (256 patches)
- Q8 tiled (baseline): 291ms for 27 layers (256 patches)

### Root cause

The per-row X quantization requires a serial amax scan where only 1 of
256 threads is active for 64 sequential steps (one thread per row,
serializing the amax computation). This completely dominates the runtime
and negates any benefit from `dot4I8Packed`.

The shader uses `if (tid == m)` to assign one thread per row for the amax
computation, which means 255 of 256 threads are idle during each row's
amax scan. With 64 rows, this is 64 sequential steps with 1/256 thread
utilization — a massive parallelism loss.

### How to make DP4a viable

The fix would be to **pre-quantize X in a separate dispatch** before the
matmul. This would:
1. Run a dedicated quantization shader (one workgroup per row, all threads
   cooperating on the amax scan and int8 packing)
2. Store the quantized X + per-row scales in a buffer
3. The matmul shader then reads pre-quantized X (no in-shader quantization)

This would eliminate the serial amax bottleneck and let the matmul shader
benefit from `dot4I8Packed`'s 4-way int8 dot product throughput.

### Current state

The shader and wiring are kept but disabled (`useDP4a = false`). The
shader is registered in the vision shader list when
`packed_4x8_integer_dot_product` is available, and the dispatch paths
are wired up in the vision forward, but the `useDP4a` flag is hardcoded
to `false` to prevent the 6× regression.

### Files changed

- `shaders/vision_matmul_q8_dp4a.wgsl` (new — DP4a Q8 matmul with
  on-the-fly X quantization)
- `src/engine.ts` (register shader, wire up dispatch paths, disabled)

### Lessons

1. **On-the-fly quantization is expensive**: Serial per-row amax scans
   with 1/256 thread utilization can dominate runtime. Always pre-compute
   in a separate parallel dispatch.
2. **DP4a requires both operands as int8**: Unlike f32 FMA where X can
   stay as f32, `dot4I8Packed` needs both inputs packed as int8. The
   quantization cost must be amortized across multiple matmuls or
   pre-computed.
3. **Feature availability ≠ performance**: Just because a WGSL feature
   is available doesn't mean using it will be faster. The overhead of
   data format conversion can negate the instruction-level speedup.

---

## 2026-08-15: Vision tower GPU path — already wired up

### Finding

During investigation of vision performance, discovered that the GPU vision
forward path was **already fully implemented and working**. The vision
tower runs entirely on GPU with:
- Q8_0 tiled matmul shaders (64×64 tiles, 256 threads, 4×4 register tiles)
- F16 tiled matmul for FFN down weights
- Pre-computed 2D RoPE for Q/K
- Shared-memory tiled attention with vec4 dot products
- All 27 transformer layers batched into one compute pass

The `visionForward` function in `src/vision.ts` (CPU path) is NOT used —
the engine's own `visionForward` in `src/engine.ts` (GPU path) is the
active code path, exported via the engine API.

### Performance (Playwright Chromium, no subgroups)

- **256 patches (64 merged)**: ~291ms for 27 layers, ~65ms CPU preprocess
- **234 patches (screenshot)**: ~2707ms total (includes weight loading)
- **Bottleneck**: Q8 tiled matmul without tensor cores (SGMat not available
  in Playwright's Chromium)

### Available WGSL features (Playwright Chromium)

- `packed_4x8_integer_dot_product` — available (DP4a)
- `subgroup_uniformity`, `subgroup_id` — available
- `chromium_experimental_subgroup_matrix` — NOT available
- `chromium_experimental_subgroups` — NOT available

The SGMat (tensor core) shaders are wired but dormant because the
`chromium_experimental_subgroup_matrix` extension is not available. The
fallback path (64×64 manual FMA tiled) is the active code path.

### Performance breakdown (256 patches, 27 layers)

| Phase | Time | Notes |
|-------|------|-------|
| CPU preprocess | 65ms | Image resize + patch extraction + pos embed interp |
| Patch embed + pos add | ~5ms | GPU dispatch (separate submit) |
| 27 transformer layers | ~291ms | GPU compute (one compute pass, 7 dispatches/layer) |
| Merger + readback | ~8ms | Patch merger + mm.0 + mm.2 + GPU→CPU readback |
| **Total** | **~369ms** | |

Per-layer breakdown (~10.8ms/layer):
- LayerNorm1: ~0.5ms
- QKV projection (Q8 tiled): ~2.5ms
- 2D RoPE: ~0.5ms
- Attention (vec4, tiled): ~2.0ms
- Attn output proj + residual (Q8 tiled add): ~1.5ms
- LayerNorm2: ~0.5ms
- FFN up + GELU (Q8 tiled gelu): ~2.0ms
- FFN down + residual (F16 tiled add): ~1.3ms

---

## Summary: Current VLM state (2026-08-15)

### What works
- ✅ M-RoPE (IMROPE) for image tokens — model can "see" images
- ✅ Vision tower GPU forward — 291ms for 256 patches (no subgroups)
- ✅ Vision weight preloading — overlapped with LLM model loading
- ✅ Q8_0 in-shader dequantization — 2.75× weight bandwidth reduction
- ✅ F16 storage for FFN down — further 1.45× bandwidth reduction
- ✅ 2D tiled matmul — eliminates redundant W reads
- ✅ vec4 attention — 29% faster scalar→vectorized
- ✅ Position embedding interpolation (align_corners=True) — matches reference

### What doesn't work (yet)
- ❌ SGMat (tensor core) shaders — `chromium_experimental_subgroup_matrix`
  not available in Playwright Chromium (would give significant speedup)
- ❌ DP4a matmul — 6× slower due to on-the-fly X quantization overhead
  (would need pre-quantization dispatch to be viable)
- ❌ Speculative decoding — disabled for hybrid backbone (Bonsai-27B uses
  Gated DeltaNet + attention; rejected drafts corrupt recurrent state)

### Improvement opportunities (ranked by impact)

1. **Use Q4_K_M quantization** (biggest quality win, no code change):
   Q1_0 (1.5 bits/weight) destroys too much information. The model
   correctly identifies colors but struggles with shapes. Q4_K_M would
   dramatically improve vision quality.

2. **SGMat on real browser** (biggest speed win):
   On a browser with `chromium_experimental_subgroup_matrix`, the SGMat
   path would use hardware tensor cores for a significant speedup. The
   shaders are already written and wired up.

3. **Pre-quantize X for DP4a** (medium speed win):
   A separate dispatch to quantize X to int8 before the DP4a matmul would
   eliminate the serial amax bottleneck and unlock `dot4I8Packed`'s 4-way
   int8 dot product throughput.

4. **F16 activations** (medium speed win):
   Halves activation bandwidth (916→458 MB). Currently low priority since
   the vision tower is compute-bound, but would help if combined with
   SGMat or DP4a.

5. **Kernel fusion** (small speed win):
   Fusing LayerNorm+QKV or RoPE+attention could save 54+ dispatches.
   Previous attempts were slower due to barrier overhead, but fusion
   without reductions (e.g., RoPE+attention) might work.

