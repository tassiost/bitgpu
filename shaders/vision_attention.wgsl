// Vision bidirectional attention with PRE-APPLIED RoPE.
// K and Q already have RoPE applied (by vision_apply_rope shader).
// This shader does pure attention: softmax(Q·K^T) · V with online softmax
// and shared-memory tiling for K/V.
//
// vec4 optimization: head_dim=72 = 4×18, so all loops use vec4 dot products
// and vec4 accumulate/add. This reduces inner loop iterations 4×.
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
const TILE_SIZE: u32 = 16u;  // 16 × 72 × 4 = 4608 bytes per tile, 9216 total < 16KB default
const HD4: u32 = 18u;        // head_dim / 4 = 72 / 4 = 18

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
// Stored as vec4 for efficient vec4 dot products (72/4 = 18 vec4 elements per row)
var<workgroup> shared_k: array<vec4<f32>, TILE_SIZE * HD4>;
var<workgroup> shared_v: array<vec4<f32>, TILE_SIZE * HD4>;

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

  // Load Q into registers as vec4 array (18 vec4 = 72 floats)
  var q_vec: array<vec4<f32>, HD4>;
  if (seg_valid && head_valid && q_valid) {
    for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
      let b = q_base + d4 * 4u;
      q_vec[d4] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]);
    }
  }

  // Online softmax state
  var max_score = -3.0e38;
  var sum_exp = 0.0;
  var acc: array<vec4<f32>, HD4>;
  for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
    acc[d4] = vec4<f32>(0.0);
  }

  // Process K/V in tiles of TILE_SIZE
  for (var tile_start = 0u; tile_start < seg_len; tile_start = tile_start + TILE_SIZE) {
    let tile_len = min(TILE_SIZE, seg_len - tile_start);

    // Cooperative load: each thread loads one K and one V vector as vec4
    if (lid.x < tile_len) {
      let kv_global = seg_start + tile_start + lid.x;
      let kv_base = kv_global * p.num_heads * p.head_dim + head * p.head_dim;
      for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
        let b = kv_base + d4 * 4u;
        shared_k[lid.x * HD4 + d4] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]);
        shared_v[lid.x * HD4 + d4] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]);
      }
    }
    workgroupBarrier();

    // Compute attention scores and accumulate V (single pass)
    if (seg_valid && head_valid && q_valid) {
      for (var t = 0u; t < tile_len; t = t + 1u) {
        // Q · K dot product using vec4 (18 dot products instead of 72 scalar multiply-adds)
        var score = 0.0;
        for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
          score = score + dot(q_vec[d4], shared_k[t * HD4 + d4]);
        }
        score = score * p.scale;

        // Online softmax update (flash-attention)
        let old_max = max_score;
        max_score = max(max_score, score);
        let correction = exp(old_max - max_score);
        let weight = exp(score - max_score);
        sum_exp = sum_exp * correction + weight;
        // V accumulation with vec4 (18 vec4 multiply-adds instead of 72 scalar)
        for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
          acc[d4] = acc[d4] * correction + weight * shared_v[t * HD4 + d4];
        }
      }
    }

    workgroupBarrier();
  }

  // Normalize and write output using vec4
  if (seg_valid && head_valid && q_valid) {
    let inv_sum = 1.0 / sum_exp;
    for (var d4 = 0u; d4 < HD4; d4 = d4 + 1u) {
      let o = acc[d4] * inv_sum;
      let b = q_base + d4 * 4u;
      out[b] = o.x;
      out[b + 1u] = o.y;
      out[b + 2u] = o.z;
      out[b + 3u] = o.w;
    }
  }
}
