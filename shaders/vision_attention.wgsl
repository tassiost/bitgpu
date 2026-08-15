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
    { let b = q_base + 0u * 4u; q_vec[0u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 1u * 4u; q_vec[1u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 2u * 4u; q_vec[2u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 3u * 4u; q_vec[3u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 4u * 4u; q_vec[4u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 5u * 4u; q_vec[5u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 6u * 4u; q_vec[6u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 7u * 4u; q_vec[7u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 8u * 4u; q_vec[8u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 9u * 4u; q_vec[9u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 10u * 4u; q_vec[10u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 11u * 4u; q_vec[11u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 12u * 4u; q_vec[12u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 13u * 4u; q_vec[13u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 14u * 4u; q_vec[14u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 15u * 4u; q_vec[15u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 16u * 4u; q_vec[16u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
    { let b = q_base + 17u * 4u; q_vec[17u] = vec4<f32>(q[b], q[b + 1u], q[b + 2u], q[b + 3u]); }
  }

  // Online softmax state
  var max_score = -3.0e38;
  var sum_exp = 0.0;
  var acc: array<vec4<f32>, HD4>;
  acc[0u] = vec4<f32>(0.0);
  acc[1u] = vec4<f32>(0.0);
  acc[2u] = vec4<f32>(0.0);
  acc[3u] = vec4<f32>(0.0);
  acc[4u] = vec4<f32>(0.0);
  acc[5u] = vec4<f32>(0.0);
  acc[6u] = vec4<f32>(0.0);
  acc[7u] = vec4<f32>(0.0);
  acc[8u] = vec4<f32>(0.0);
  acc[9u] = vec4<f32>(0.0);
  acc[10u] = vec4<f32>(0.0);
  acc[11u] = vec4<f32>(0.0);
  acc[12u] = vec4<f32>(0.0);
  acc[13u] = vec4<f32>(0.0);
  acc[14u] = vec4<f32>(0.0);
  acc[15u] = vec4<f32>(0.0);
  acc[16u] = vec4<f32>(0.0);
  acc[17u] = vec4<f32>(0.0);


  // Process K/V in tiles of TILE_SIZE
  for (var tile_start = 0u; tile_start < seg_len; tile_start = tile_start + TILE_SIZE) {
    let tile_len = min(TILE_SIZE, seg_len - tile_start);

    // Cooperative load: each thread loads one K and one V vector as vec4
    if (lid.x < tile_len) {
      let kv_global = seg_start + tile_start + lid.x;
      let kv_base = kv_global * p.num_heads * p.head_dim + head * p.head_dim;
      { let b = kv_base + 0u * 4u; shared_k[lid.x * HD4 + 0u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 0u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 1u * 4u; shared_k[lid.x * HD4 + 1u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 1u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 2u * 4u; shared_k[lid.x * HD4 + 2u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 2u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 3u * 4u; shared_k[lid.x * HD4 + 3u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 3u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 4u * 4u; shared_k[lid.x * HD4 + 4u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 4u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 5u * 4u; shared_k[lid.x * HD4 + 5u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 5u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 6u * 4u; shared_k[lid.x * HD4 + 6u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 6u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 7u * 4u; shared_k[lid.x * HD4 + 7u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 7u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 8u * 4u; shared_k[lid.x * HD4 + 8u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 8u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 9u * 4u; shared_k[lid.x * HD4 + 9u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 9u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 10u * 4u; shared_k[lid.x * HD4 + 10u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 10u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 11u * 4u; shared_k[lid.x * HD4 + 11u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 11u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 12u * 4u; shared_k[lid.x * HD4 + 12u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 12u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 13u * 4u; shared_k[lid.x * HD4 + 13u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 13u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 14u * 4u; shared_k[lid.x * HD4 + 14u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 14u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 15u * 4u; shared_k[lid.x * HD4 + 15u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 15u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 16u * 4u; shared_k[lid.x * HD4 + 16u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 16u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }
      { let b = kv_base + 17u * 4u; shared_k[lid.x * HD4 + 17u] = vec4<f32>(k[b], k[b + 1u], k[b + 2u], k[b + 3u]); shared_v[lid.x * HD4 + 17u] = vec4<f32>(v[b], v[b + 1u], v[b + 2u], v[b + 3u]); }

    }
    workgroupBarrier();

    // Compute attention scores and accumulate V (single pass)
    if (seg_valid && head_valid && q_valid) {
      for (var t = 0u; t < tile_len; t = t + 1u) {
        // Q · K dot product using vec4 (18 dot products instead of 72 scalar multiply-adds)
        var score = 0.0;
        score = score + dot(q_vec[0u], shared_k[t * HD4 + 0u]);
        score = score + dot(q_vec[1u], shared_k[t * HD4 + 1u]);
        score = score + dot(q_vec[2u], shared_k[t * HD4 + 2u]);
        score = score + dot(q_vec[3u], shared_k[t * HD4 + 3u]);
        score = score + dot(q_vec[4u], shared_k[t * HD4 + 4u]);
        score = score + dot(q_vec[5u], shared_k[t * HD4 + 5u]);
        score = score + dot(q_vec[6u], shared_k[t * HD4 + 6u]);
        score = score + dot(q_vec[7u], shared_k[t * HD4 + 7u]);
        score = score + dot(q_vec[8u], shared_k[t * HD4 + 8u]);
        score = score + dot(q_vec[9u], shared_k[t * HD4 + 9u]);
        score = score + dot(q_vec[10u], shared_k[t * HD4 + 10u]);
        score = score + dot(q_vec[11u], shared_k[t * HD4 + 11u]);
        score = score + dot(q_vec[12u], shared_k[t * HD4 + 12u]);
        score = score + dot(q_vec[13u], shared_k[t * HD4 + 13u]);
        score = score + dot(q_vec[14u], shared_k[t * HD4 + 14u]);
        score = score + dot(q_vec[15u], shared_k[t * HD4 + 15u]);
        score = score + dot(q_vec[16u], shared_k[t * HD4 + 16u]);
        score = score + dot(q_vec[17u], shared_k[t * HD4 + 17u]);

        score = score * p.scale;

        // Online softmax update (flash-attention)
        let old_max = max_score;
        max_score = max(max_score, score);
        let correction = exp(old_max - max_score);
        let weight = exp(score - max_score);
        sum_exp = sum_exp * correction + weight;
        // V accumulation with vec4 (18 vec4 multiply-adds instead of 72 scalar)
        acc[0u] = acc[0u] * correction + weight * shared_v[t * HD4 + 0u];
        acc[1u] = acc[1u] * correction + weight * shared_v[t * HD4 + 1u];
        acc[2u] = acc[2u] * correction + weight * shared_v[t * HD4 + 2u];
        acc[3u] = acc[3u] * correction + weight * shared_v[t * HD4 + 3u];
        acc[4u] = acc[4u] * correction + weight * shared_v[t * HD4 + 4u];
        acc[5u] = acc[5u] * correction + weight * shared_v[t * HD4 + 5u];
        acc[6u] = acc[6u] * correction + weight * shared_v[t * HD4 + 6u];
        acc[7u] = acc[7u] * correction + weight * shared_v[t * HD4 + 7u];
        acc[8u] = acc[8u] * correction + weight * shared_v[t * HD4 + 8u];
        acc[9u] = acc[9u] * correction + weight * shared_v[t * HD4 + 9u];
        acc[10u] = acc[10u] * correction + weight * shared_v[t * HD4 + 10u];
        acc[11u] = acc[11u] * correction + weight * shared_v[t * HD4 + 11u];
        acc[12u] = acc[12u] * correction + weight * shared_v[t * HD4 + 12u];
        acc[13u] = acc[13u] * correction + weight * shared_v[t * HD4 + 13u];
        acc[14u] = acc[14u] * correction + weight * shared_v[t * HD4 + 14u];
        acc[15u] = acc[15u] * correction + weight * shared_v[t * HD4 + 15u];
        acc[16u] = acc[16u] * correction + weight * shared_v[t * HD4 + 16u];
        acc[17u] = acc[17u] * correction + weight * shared_v[t * HD4 + 17u];

      }
    }

    workgroupBarrier();
  }

  // Normalize and write output using vec4
  if (seg_valid && head_valid && q_valid) {
    let inv_sum = 1.0 / sum_exp;
    { let o = acc[0u] * inv_sum; let b = q_base + 0u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[1u] * inv_sum; let b = q_base + 1u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[2u] * inv_sum; let b = q_base + 2u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[3u] * inv_sum; let b = q_base + 3u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[4u] * inv_sum; let b = q_base + 4u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[5u] * inv_sum; let b = q_base + 5u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[6u] * inv_sum; let b = q_base + 6u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[7u] * inv_sum; let b = q_base + 7u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[8u] * inv_sum; let b = q_base + 8u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[9u] * inv_sum; let b = q_base + 9u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[10u] * inv_sum; let b = q_base + 10u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[11u] * inv_sum; let b = q_base + 11u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[12u] * inv_sum; let b = q_base + 12u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[13u] * inv_sum; let b = q_base + 13u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[14u] * inv_sum; let b = q_base + 14u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[15u] * inv_sum; let b = q_base + 15u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[16u] * inv_sum; let b = q_base + 16u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }
    { let o = acc[17u] * inv_sum; let b = q_base + 17u * 4u; out[b] = o.x; out[b + 1u] = o.y; out[b + 2u] = o.z; out[b + 3u] = o.w; }

  }
}
