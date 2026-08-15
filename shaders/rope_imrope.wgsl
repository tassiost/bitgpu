// IMROPE: Interleaved M-RoPE for Qwen3-VL multimodal models.
// Rotates pairs of adjacent dimensions (GPT-J style) instead of halves (NEOX).
// cos/sin are [S, ROT] where each pair (2j, 2j+1) shares the same value (expanded from [S, ROT/2]).
// The M-RoPE section positions (temporal/height/width) are baked into the cos/sin cache on the CPU.
// x/y are [S, H, D]; only the first ROT dims of each head are rotated, the rest pass through.
struct Params { S: u32, H: u32, D: u32, ROT: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> cosb: array<f32>;     // [S, ROT]
@group(0) @binding(3) var<storage, read> sinb: array<f32>;     // [S, ROT]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  if (d >= p.ROT) { y[idx] = x[idx]; return; }   // passthrough tail
  let sh = idx / p.D;
  let s = sh / p.H;
  // Interleaved rotation (GPT-J style): pair (d_even, d_odd) = (2j, 2j+1)
  // y[even] = x[even] * cos - x[odd] * sin
  // y[odd]  = x[even] * sin + x[odd] * cos
  let cos_val = cosb[s * p.ROT + d];
  let sin_val = sinb[s * p.ROT + d];
  var rot: f32;
  if (d % 2u == 0u) {
    rot = -x[idx + 1u];  // pair with next dimension
  } else {
    rot = x[idx - 1u];   // pair with previous dimension
  }
  y[idx] = x[idx] * cos_val + rot * sin_val;
}
