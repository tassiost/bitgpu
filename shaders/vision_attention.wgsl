// Vision bidirectional attention with 2D RoPE: full self-attention within each image segment.
// Uses cu_seqlens to handle variable-length images in a batch.
// E.g., cu_seqlens = [0, 196, 392] means image 1 has 196 patches, image 2 has 196.
// Attention is applied within each segment independently (no cross-image attention).
//
// 2D RoPE (Qwen3-VL style):
//   rotary_dim = head_dim // 2 (partial rotary, 50%)
//   cos/sin are precomputed [total_seq, head_dim] (cat(freqs, freqs) layout)
//   rotate_half(x) = [-x[rotary_dim:], x[:rotary_dim]]  (NeoX style)
//   q_rot = q * cos + rotate_half(q) * sin
//
// This is the online softmax attention (flash-attention style): one workgroup
// per (head, query_block), with 32 threads per workgroup — each thread handles
// one query independently. No causal mask — bidirectional.
//
// head_dim = 72, num_heads = 16, rotary_dim = 36
// scaling = head_dim^-0.5 = 72^-0.5 ≈ 0.1179
//
// Performance: @workgroup_size(32) fills a full GPU wavefront (32 threads),
// giving ~32x better occupancy than @workgroup_size(1). Each thread does the
// full online softmax loop over all keys for its assigned query.

const WG_SIZE: u32 = 32u;

struct Params {
  num_heads: u32,     // 16
  head_dim: u32,      // 72
  num_segments: u32,  // number of images (segments)
  scale: f32,         // 1/sqrt(head_dim)
  rotary_dim: u32,    // head_dim // 2 = 36
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;          // [total_seq, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> k: array<f32>;          // [total_seq, num_heads, head_dim]
@group(0) @binding(3) var<storage, read> v: array<f32>;          // [total_seq, num_heads, head_dim]
@group(0) @binding(4) var<storage, read> cu_seqlens: array<u32>; // [num_segments + 1]
@group(0) @binding(5) var<storage, read> cos_buf: array<f32>;    // [total_seq, head_dim]
@group(0) @binding(6) var<storage, read> sin_buf: array<f32>;    // [total_seq, head_dim]
@group(0) @binding(7) var<storage, read_write> out: array<f32>;  // [total_seq, num_heads, head_dim]

// Apply NeoX-style RoPE to a vector in-place.
// rotate_half(x) = [-x[rotary_dim:], x[:rotary_dim]]
// result = x * cos + rotate_half(x) * sin
fn apply_rope(vec: ptr<function, array<f32, 72>>, cos_sin_base: u32) {
  let rd = p.rotary_dim;
  for (var d = 0u; d < p.head_dim; d++) {
    let c = cos_buf[cos_sin_base + d];
    let s = sin_buf[cos_sin_base + d];
    var rotated: f32;
    if (d < rd) {
      rotated = -(*vec)[d + rd];
    } else {
      rotated = (*vec)[d - rd];
    }
    (*vec)[d] = (*vec)[d] * c + rotated * s;
  }
}

@compute @workgroup_size(32)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  // Each workgroup handles a block of 32 queries for one (segment, head).
  // Each thread handles one query independently — no inter-thread communication needed.
  let seg = wg.x;        // segment index (image)
  let head = wg.y;       // head index
  let q_local = wg.z * WG_SIZE + lid.x;  // query index within segment

  if (seg >= p.num_segments || head >= p.num_heads) { return; }

  let seg_start = cu_seqlens[seg];
  let seg_end = cu_seqlens[seg + 1u];
  let seg_len = seg_end - seg_start;

  if (q_local >= seg_len) { return; }

  let q_global = seg_start + q_local;
  let q_base = q_global * p.num_heads * p.head_dim + head * p.head_dim;
  let rope_base = q_global * p.head_dim;  // cos/sin are [total_seq, head_dim] (no heads dim)

  // Load Q and apply RoPE
  var q_vec: array<f32, 72>;
  for (var d = 0u; d < p.head_dim; d++) {
    q_vec[d] = q[q_base + d];
  }
  apply_rope(&q_vec, rope_base);

  // Online softmax attention (flash-attention style)
  var max_score = -3.0e38;
  var sum_exp = 0.0;
  var acc: array<f32, 72>;
  for (var d = 0u; d < p.head_dim; d++) {
    acc[d] = 0.0;
  }

  // First pass: compute max score and accumulate
  for (var kv_local = 0u; kv_local < seg_len; kv_local++) {
    let kv_global = seg_start + kv_local;
    let k_base = kv_global * p.num_heads * p.head_dim + head * p.head_dim;
    let k_rope_base = kv_global * p.head_dim;

    // Load K and apply RoPE
    var k_vec: array<f32, 72>;
    for (var d = 0u; d < p.head_dim; d++) {
      k_vec[d] = k[k_base + d];
    }
    apply_rope(&k_vec, k_rope_base);

    // Dot product Q · K (both RoPE-applied)
    var score = 0.0;
    for (var d = 0u; d < p.head_dim; d++) {
      score = score + q_vec[d] * k_vec[d];
    }
    score = score * p.scale;

    // Online softmax update
    let old_max = max_score;
    max_score = max(max_score, score);
    let correction = exp(old_max - max_score);
    sum_exp = sum_exp * correction + exp(score - max_score);
    for (var d = 0u; d < p.head_dim; d++) {
      acc[d] = acc[d] * correction + v[k_base + d] * exp(score - max_score);
    }
  }

  // Normalize and write output
  let inv_sum = 1.0 / sum_exp;
  for (var d = 0u; d < p.head_dim; d++) {
    out[q_base + d] = acc[d] * inv_sum;
  }
}
