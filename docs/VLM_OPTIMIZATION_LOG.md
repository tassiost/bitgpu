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

**Performance results** (234-patch screenshot, M3 Mac, cold system):

| Metric | Baseline | After opt | Improvement |
|--------|----------|-----------|-------------|
| Vision tower | 50.8s | 33.9s | 33% faster |
| Prefill | 18.2s | 11.5s | 37% faster |
| Tokens/s | 3.6 | 5.7 | 58% faster |

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

**Bandwidth analysis**: The vision tower is memory-bandwidth bound, not
compute bound. Total weight reads per layer: 55.5 MB (f32). At ~100 GB/s
(M3 unified memory), theoretical minimum is ~15ms/layer, but actual is
~1260ms/layer — 84× slower. The weights are dequantized from Q8_0 to f32
on CPU, then uploaded as f32 buffers. Keeping them as Q8_0 and dequantizing
in-shader would reduce weight bandwidth 4× (P3 below).

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

### 4. Previous optimizations (commit e177133)

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

### P3: Q8_0 in-shader dequantization (BIGGEST WIN)

**Analysis**: The vision tower is memory-bandwidth bound, not compute bound.
Total weight reads per layer: 55.5 MB (f32). At ~100 GB/s (M3 unified memory),
theoretical minimum is ~15ms per layer, but actual is ~1260ms — 84× slower.

The weights are currently dequantized from Q8_0 to f32 on CPU during load,
then uploaded as f32 buffers. If we kept them as Q8_0 and dequantized in the
shader, weight buffers would be 4× smaller:
- QKV: 15.9 MB → 4.2 MB
- FFN up: 19.8 MB → 5.2 MB
- FFN down: 19.8 MB → 5.2 MB
- Total per layer: 55.5 MB → 14.6 MB
- Total 27 layers: 1.5 GB → 394 MB

This would dramatically reduce memory bandwidth pressure. Each Q8_0 block
is 34 bytes (2-byte f16 scale + 32 int8 values). The shader reads the scale,
then multiplies each int8 value by it.

**Challenge**: Requires rewriting all matmul shaders to handle Q8_0 layout.
The weight buffer binding changes from `array<f32>` to `array<u32>` (packed
Q8_0 blocks). Each thread reads its Q8_0 block, dequantizes to f32, then
does the dot product.

### P4: f16 storage for vision activations

All activation buffers use f32 (4 bytes). Using f16 would halve memory
bandwidth for activations. Apple Silicon has native f16 support (Metal).
Requires `shader-f16` feature. Vision tower activations can tolerate f16
precision (CLIP ViT models run successfully with f16 on WebGPU).

**Note**: Weight bandwidth (55.5 MB/layer) dominates activation bandwidth
(~9.4 MB/layer), so P3 (Q8_0 dequant) is more impactful than P4 (f16
activations). Doing both would be ideal.

### P5: 2D tiling for matmul

Current matmul: 1 thread per output column, loops over K. Each workgroup
(64 threads) handles 64 columns. For QKV (3456 outputs), that's 54 workgroups.

2D tiling (from llama.cpp): each thread computes a 4×4 tile of outputs.
TILE_M=4, TILE_N=4, WG_SIZE=8×8=64. This improves weight reuse and reduces
workgroup count. Research shows 79.5% of optimal GFLOPs with this pattern.

### P6: Subgroup operations (Chrome 134+)

`subgroupAdd()`, `subgroupMax()` for softmax reductions. Eliminates shared
memory and barriers. 1.29x prefill speedup in ONNX Runtime tests.

**Status**: Chrome 134+ only. Not in Safari or Firefox yet. Must detect and
fall back. Apple Silicon supports subgroups in Metal but Safari doesn't
expose them in WebGPU yet.

### P7: Create test images with known text content
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
