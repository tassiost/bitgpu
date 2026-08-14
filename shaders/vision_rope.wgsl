// Vision bidirectional self-attention with 2D RoPE.
// Unlike the LLM's causal attention, the vision tower uses FULL bidirectional
// attention — every patch attends to every other patch in the same image.
//
// This shader applies 2D RoPE to Q and K. The attention itself uses a segmented
// attention kernel (cu_seqlens for batched images).
//
// 2D RoPE: position_ids are (temporal, height, width). The head_dim is split
// into sections: [temporal_dims, height_dims, width_dims] (mrope_section).
// Each section gets its own frequency applied based on the corresponding
// position coordinate.
//
// mrope_section for vision: [24, 20, 20] sums to 64, but head_dim=72 so
// head_dim/2=36. The vision tower uses its own simpler 2D RoPE with dim=36.
// Sections: [12, 12, 12] (temporal, height, width) for dim=36.

struct Params {
  seq_len: u32,       // total sequence length (all patches)
  num_heads: u32,     // 16
  head_dim: u32,      // 72
  rot_dim: u32,       // 36 (= head_dim / 2 for vision RoPE)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;          // [seq, num_heads, head_dim]
@group(0) @binding(2) var<storage, read> k: array<f32>;          // [seq, num_heads, head_dim]
@group(0) @binding(3) var<storage, read> cos: array<f32>;        // [seq, rot_dim]
@group(0) @binding(4) var<storage, read> sin: array<f32>;        // [seq, rot_dim]
@group(0) @binding(5) var<storage, read_write> q_out: array<f32>;
@group(0) @binding(6) var<storage, read_write> k_out: array<f32>;

@compute @workgroup_size(64)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>,
) {
  let total = p.seq_len * p.num_heads * p.rot_dim;
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= total) { return; }

  // Decode: (seq_pos, head, rot_idx)
  let rot_idx = idx % p.rot_dim;
  let head = (idx / p.rot_dim) % p.num_heads;
  let seq_pos = idx / (p.rot_dim * p.num_heads);

  let cos_val = cos[seq_pos * p.rot_dim + rot_idx];
  let sin_val = sin[seq_pos * p.rot_dim + rot_idx];

  // Each rot_idx rotates a pair: (i, i + rot_dim) in the head_dim
  let base = seq_pos * p.num_heads * p.head_dim + head * p.head_dim;
  let i1 = rot_idx;
  let i2 = rot_idx + p.rot_dim;

  let q1 = q[base + i1];
  let q2 = q[base + i2];
  q_out[base + i1] = q1 * cos_val - q2 * sin_val;
  q_out[base + i2] = q1 * sin_val + q2 * cos_val;

  let k1 = k[base + i1];
  let k2 = k[base + i2];
  k_out[base + i1] = k1 * cos_val - k2 * sin_val;
  k_out[base + i2] = k1 * sin_val + k2 * cos_val;
}
