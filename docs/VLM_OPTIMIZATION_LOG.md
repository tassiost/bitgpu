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

**Performance results** (234-patch screenshot, M3 Mac):

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

---

## Future Optimization Opportunities

### P3: Use f16 storage for vision activations
All activation buffers use f32 (4 bytes). Using f16 would halve memory
bandwidth. The vision tower is memory-bandwidth bound. Need to verify
accuracy impact — LayerNorm should keep values in reasonable range.

### P4: Subgroup operations for LayerNorm
LayerNorm shader uses @workgroup_size(64) but only thread 0 does the
reduction (3 sequential passes over D=1152). Use subgroupAdd() for
mean/variance reductions — 64x more parallelism.

### P5: Fuse LayerNorm into QKV matmul
LayerNorm and QKV matmul are separate dispatches. Fusing them eliminates
one full read+write of hidden states per layer (27 × numPatches × 1152 × 4
bytes saved).

### P6: Request higher workgroup storage limit
Current TILE_SIZE=16 due to 16KB default workgroup storage limit. The
adapter supports 32KB — requesting it in `requiredLimits` would allow
TILE_SIZE=32, doubling attention tile efficiency.

### P7: Create test images with known text content
Create synthetic images with clear, large text to verify text recognition
accuracy. This will help measure the impact of the bicubic fix and future
accuracy improvements.

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
