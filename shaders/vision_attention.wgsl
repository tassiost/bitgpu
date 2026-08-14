// Vision bidirectional attention with PRE-APPLIED RoPE.
// K and Q already have RoPE applied (by vision_apply_rope shader).
// This shader does pure attention: softmax(Q·K^T) · V with online softmax
// and shared-memory tiling for K/V.
//
// Tiling: K/V are loaded cooperatively into workgroup shared memory.
// Each workgroup of 32 threads loads 32 K/V vectors at a time (one per
// thread), then all threads iterate over the shared tile.
//
// Single-pass online softmax (flash-attention style): scores are computed
// once, used immediately for both running max/sum update and V accumulation.
//
// head_dim = 72, num_heads = 16, scale = 72^-0.5 ≈ 0.1179
//
// NOTE: No early return before workgroupBarrier — uses `valid` flag to
// ensure all threads reach barriers uniformly.

const WG_SIZE: u32 = 32u;
const TILE_SIZE: u32 = 16u;  // 16 × 72 × 4 = 4608 bytes per tile, 9216 total < 16KB limit

struct Params {
  num_heads: u32,     // 16
  head_dim: u32,      // 72
  num_segments: u32,  // number of image segments
  scale: f32,         // 1/sqrt(head_dim)
  _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;          // [total_seq, num_heads, head_dim] (RoPE applied)
@group(0) @binding(2) var<storage, read> k: array<f32>;          // [total_seq, num_heads, head_dim] (RoPE applied)
@group(0) @binding(3) var<storage, read> v: array<f32>;          // [total_seq, num_heads, head_dim]
@group(0) @binding(4) var<storage, read> cu_seqlens: array<u32>; // [num_segments + 1]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;  // [total_seq, num_heads, head_dim]

// Shared memory tiles: [TILE_SIZE, head_dim] = 32 × 72 × 4 = 9216 bytes each
var<workgroup> shared_k: array<f32, TILE_SIZE * 72>;
var<workgroup> shared_v: array<f32, TILE_SIZE * 72>;

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let seg = wg.x;
  let head = wg.y;
  let q_local = wg.z * WG_SIZE + lid.x;

  // Uniform guard: seg and head come from workgroup_id (uniform)
  let seg_valid = seg < p.num_segments;
  let head_valid = head < p.num_heads;

  let seg_start = cu_seqlens[seg];
  let seg_end = cu_seqlens[seg + 1u];
  let seg_len = seg_end - seg_start;

  // q_local is non-uniform (depends on lid.x), so use a flag
  let q_valid = q_local < seg_len;

  let q_global = seg_start + q_local;
  let q_base = q_global * p.num_heads * p.head_dim + head * p.head_dim;

  // Load Q into registers (only valid threads need it, but all threads
  // must participate in barriers below)
  var q_vec: array<f32, 72>;
  if (seg_valid && head_valid && q_valid) {
    for (var d = 0u; d < p.head_dim; d++) {
      q_vec[d] = q[q_base + d];
    }
  }

  // Online softmax state
  var max_score = -3.0e38;
  var sum_exp = 0.0;
  var acc: array<f32, 72>;
  for (var d = 0u; d < p.head_dim; d++) {
    acc[d] = 0.0;
  }

  // Process K/V in tiles of TILE_SIZE
  for (var tile_start = 0u; tile_start < seg_len; tile_start = tile_start + TILE_SIZE) {
    let tile_len = min(TILE_SIZE, seg_len - tile_start);

    // Cooperative load: each thread loads one K and one V vector
    if (lid.x < tile_len) {
      let kv_global = seg_start + tile_start + lid.x;
      let kv_base = kv_global * p.num_heads * p.head_dim + head * p.head_dim;
      for (var d = 0u; d < p.head_dim; d++) {
        shared_k[lid.x * 72u + d] = k[kv_base + d];
        shared_v[lid.x * 72u + d] = v[kv_base + d];
      }
    }
    workgroupBarrier();

    // Compute attention scores and accumulate V (single pass)
    if (seg_valid && head_valid && q_valid) {
      for (var t = 0u; t < tile_len; t++) {
        // Q · K dot product (both already RoPE-applied)
        var score = 0.0;
        for (var d = 0u; d < p.head_dim; d++) {
          score = score + q_vec[d] * shared_k[t * 72u + d];
        }
        score = score * p.scale;

        // Online softmax update (flash-attention)
        let old_max = max_score;
        max_score = max(max_score, score);
        let correction = exp(old_max - max_score);
        let weight = exp(score - max_score);
        sum_exp = sum_exp * correction + weight;
        for (var d = 0u; d < p.head_dim; d++) {
          acc[d] = acc[d] * correction + weight * shared_v[t * 72u + d];
        }
      }
    }

    workgroupBarrier();
  }

  // Normalize and write output
  if (seg_valid && head_valid && q_valid) {
    let inv_sum = 1.0 / sum_exp;
    for (var d = 0u; d < p.head_dim; d++) {
      out[q_base + d] = acc[d] * inv_sum;
    }
  }
}
