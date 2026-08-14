// Apply 2D RoPE to Q and K vectors in-place (or to separate output buffers).
// One workgroup per patch; 64 threads cooperatively process all heads.
// Each thread handles one (head, dim) pair, loading the value, its paired
// rotate_half value, cos, and sin, then writing the rotated result.
//
// RoPE (NeoX-style rotate_half):
//   rotary_dim = head_dim // 2 (partial rotary, 50%)
//   rotate_half(x)[d] = (d < rotary_dim) ? -x[d + rotary_dim] : x[d - rotary_dim]
//   q_rot[d] = x[d] * cos[d] + rotate_half(x)[d] * sin[d]
//
// Layout: q/k are [num_patches, num_heads, head_dim]
//         cos/sin are [num_patches, head_dim] (shared across heads)

const ROTARY_DIM: u32 = 36u;  // head_dim // 2
const HEAD_DIM: u32 = 72u;

struct Params {
  num_patches: u32,
  num_heads: u32,
  head_dim: u32,
  rotary_dim: u32,
  _pad0: u32, _pad1: u32, _pad2: u32, _pad3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q_in: array<f32>;   // [num_patches, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> k_in: array<f32>;   // [num_patches, num_heads, head_dim]
@group(0) @binding(3) var<storage, read_write> q_out: array<f32>;  // [num_patches, num_heads, head_dim]
@group(0) @binding(4) var<storage, read_write> k_out: array<f32>;  // [num_patches, num_heads, head_dim]
@group(0) @binding(5) var<storage, read> cos_buf: array<f32>;     // [num_patches, head_dim]
@group(0) @binding(6) var<storage, read> sin_buf: array<f32>;     // [num_patches, head_dim]

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let patch_idx = wg.x;
  if (patch_idx >= p.num_patches) { return; }

  let rope_base = patch_idx * p.head_dim;
  let qk_base = patch_idx * p.num_heads * p.head_dim;

  // Each thread handles one (head, dim) pair.
  // 64 threads × iter = num_heads * head_dim = 16 * 72 = 1152 values
  let total = p.num_heads * p.head_dim;
  for (var idx = lid.x; idx < total; idx = idx + 64u) {
    let head = idx / p.head_dim;
    let d = idx % p.head_dim;
    let val_base = qk_base + head * p.head_dim;

    // Load cos/sin (shared across heads for this patch)
    let c = cos_buf[rope_base + d];
    let s = sin_buf[rope_base + d];

    // Compute rotate_half
    var q_rotated: f32;
    var k_rotated: f32;
    if (d < p.rotary_dim) {
      q_rotated = -q_in[val_base + d + p.rotary_dim];
      k_rotated = -k_in[val_base + d + p.rotary_dim];
    } else {
      q_rotated = q_in[val_base + d - p.rotary_dim];
      k_rotated = k_in[val_base + d - p.rotary_dim];
    }

    // Apply RoPE: x * cos + rotate_half(x) * sin
    q_out[val_base + d] = q_in[val_base + d] * c + q_rotated * s;
    k_out[val_base + d] = k_in[val_base + d] * c + k_rotated * s;
  }
}
